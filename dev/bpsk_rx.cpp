#include <GL/glew.h>
#include <SDL2/SDL.h>

#include <SoapySDR/Device.h>
#include <SoapySDR/Formats.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <fftw3.h>

#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_sdl2.h"
#include "imgui.h"
#include "implot.h"

struct LockFreeRingBuffer {
    float *buffer;
    int size;
    volatile int write_pos;
    volatile int read_pos;
};

struct CostasLoop {
    float phase;
    float freq;
    float alpha;
    float beta;
};

struct SharedData {
    char *sdr_uri;
    struct LockFreeRingBuffer raw_rb;
    struct LockFreeRingBuffer filtered_i_rb;
    struct LockFreeRingBuffer filtered_q_rb;
    struct LockFreeRingBuffer spectrum_rb;  // Для хранения спектра
    int running;
    float current_freq_offset;
    float current_signal_power;
    int spectrum_available;  // Флаг наличия спектра
    float last_spectrum[1024]; // Последний вычисленный спектр
};

int global_running = 1;
pthread_mutex_t fft_mutex = PTHREAD_MUTEX_INITIALIZER;

void sigint_handler(int sig) {
    (void)sig;
    printf("\nCaught SIGINT, shutting down...\n");
    global_running = 0;
}

void init_ring_buffer(struct LockFreeRingBuffer *rb, int size) {
    rb->buffer = (float*)calloc(size, sizeof(float));
    rb->size = size;
    rb->write_pos = 0;
    rb->read_pos = 0;
    printf("Initialized ring buffer of size %d\n", size);
}

int ring_buffer_write(struct LockFreeRingBuffer *rb, float value) {
    int write_pos = rb->write_pos;
    int read_pos = rb->read_pos;
    int next_write_pos = (write_pos + 1) % rb->size;
    
    if (next_write_pos == read_pos) {
        return 0; // Buffer full
    }
    
    rb->buffer[write_pos] = value;
    rb->write_pos = next_write_pos;
    return 1;
}

int ring_buffer_available(struct LockFreeRingBuffer *rb) {
    int write_pos = rb->write_pos;
    int read_pos = rb->read_pos;
    
    if (write_pos >= read_pos) {
        return write_pos - read_pos;
    } else {
        return rb->size - read_pos + write_pos;
    }
}

void ring_buffer_advance(struct LockFreeRingBuffer *rb, int count) {
    int read_pos = rb->read_pos;
    rb->read_pos = (read_pos + count) % rb->size;
}

void ring_buffer_peek(struct LockFreeRingBuffer *rb, float *dest, int start_offset, int count) {
    int read_pos = rb->read_pos;
    for (int i = 0; i < count; i++) {
        int pos = (read_pos + start_offset + i) % rb->size;
        dest[i] = rb->buffer[pos];
    }
}

float gardner_ted(struct LockFreeRingBuffer *filtered_i_rb, 
                  struct LockFreeRingBuffer *filtered_q_rb, 
                  int symbol_idx, int Nsps) {
    (void)Nsps;
    
    int read_pos = filtered_i_rb->read_pos;
    
    int early = (read_pos + symbol_idx - 1 + filtered_i_rb->size) % filtered_i_rb->size;
    int middle = (read_pos + symbol_idx) % filtered_i_rb->size;
    int late = (read_pos + symbol_idx + 1) % filtered_i_rb->size;
    
    float i_early = filtered_i_rb->buffer[early];
    float i_middle = filtered_i_rb->buffer[middle];
    float i_late = filtered_i_rb->buffer[late];
    
    float q_early = filtered_q_rb->buffer[early];
    float q_middle = filtered_q_rb->buffer[middle];
    float q_late = filtered_q_rb->buffer[late];
    
    return ((i_late - i_early) * i_middle) + ((q_late - q_early) * q_middle);
}

void apply_matched_filter_with_downsampling(struct LockFreeRingBuffer *raw_rb, 
                                           struct LockFreeRingBuffer *filtered_i_rb,
                                           struct LockFreeRingBuffer *filtered_q_rb, 
                                           int Nsps) {
    static float *h = NULL;
    static int h_initialized = 0;
    static int sample_counter = 0;
    
    if (!h_initialized) {
        h = (float*)malloc(Nsps * sizeof(float));
        for(int i = 0; i < Nsps; i++) {
            float t = (float)(i - Nsps/2) / (Nsps/2);
            if (t == 0) {
                h[i] = 1.0;
            } else {
                h[i] = sin(M_PI * t) / (M_PI * t) * cos(0.5 * M_PI * t) / (1 - 4 * t * t);
            }
        }
        float sum = 0;
        for(int i = 0; i < Nsps; i++) sum += h[i];
        for(int i = 0; i < Nsps; i++) h[i] /= sum;
        
        h_initialized = 1;
        sample_counter = 0;
        printf("Matched filter initialized with %d taps\n", Nsps);
    }
    
    int available = ring_buffer_available(raw_rb);
    if (available < Nsps * 2) return;
    
    int read_pos = raw_rb->read_pos;
    int samples_to_process = available - (available % Nsps);
    if (samples_to_process > raw_rb->size / 2) {
        samples_to_process = raw_rb->size / 2;
    }
    
    for (int n = 0; n < samples_to_process; n += 2) {
        int pos = (read_pos + n) % raw_rb->size;
        float conv_i = 0;
        float conv_q = 0;
        
        for (int k = 0; k < Nsps; k++) {
            int sample_pos = (pos - k * 2 + raw_rb->size) % raw_rb->size;
            conv_i += raw_rb->buffer[sample_pos] * h[k];
        }
        
        for (int k = 0; k < Nsps; k++) {
            int sample_pos = (pos + 1 - k * 2 + raw_rb->size) % raw_rb->size;
            conv_q += raw_rb->buffer[sample_pos] * h[k];
        }
        
        sample_counter++;
        if (sample_counter >= Nsps) {
            ring_buffer_write(filtered_i_rb, conv_i);
            ring_buffer_write(filtered_q_rb, conv_q);
            sample_counter = 0;
        }
    }
    
    raw_rb->read_pos = (read_pos + samples_to_process) % raw_rb->size;
}

void costas_loop_update(struct CostasLoop *c, float *i, float *q) {
    float cos_p = cosf(c->phase);
    float sin_p = sinf(c->phase);

    float i_rot = (*i) * cos_p + (*q) * sin_p;
    float q_rot = -(*i) * sin_p + (*q) * cos_p;

    float error = i_rot * q_rot;

    c->freq += c->beta * error;
    c->phase += c->freq + c->alpha * error;

    if (c->phase > M_PI) c->phase -= 2 * M_PI;
    if (c->phase < -M_PI) c->phase += 2 * M_PI;

    *i = i_rot;
    *q = q_rot;
}

// Исправленная функция для вычисления спектра
void compute_spectrum(float *samples, int count, float *spectrum_out, int n_fft) {
    if(count < n_fft * 2) {
        printf("Warning: Not enough samples for FFT (%d < %d)\n", count, n_fft * 2);
        return;
    }
    
    pthread_mutex_lock(&fft_mutex);
    
    static fftw_complex *in = NULL;
    static fftw_complex *out = NULL;
    static fftw_plan plan = NULL;
    static int last_n_fft = 0;
    
    // Инициализация FFTW при первом вызове или изменении размера
    if(last_n_fft != n_fft) {
        if(plan) {
            fftw_destroy_plan(plan);
            plan = NULL;
        }
        if(in) fftw_free(in);
        if(out) fftw_free(out);
        
        in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_fft);
        out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_fft);
        
        if(!in || !out) {
            printf("FFTW: Failed to allocate memory\n");
            pthread_mutex_unlock(&fft_mutex);
            return;
        }
        
        plan = fftw_plan_dft_1d(n_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        if(!plan) {
            printf("FFTW: Failed to create plan\n");
            pthread_mutex_unlock(&fft_mutex);
            return;
        }
        
        last_n_fft = n_fft;
        printf("FFTW: Initialized with N=%d\n", n_fft);
    }
    
    // Заполняем входной массив
    for(int i = 0; i < n_fft; i++) {
        in[i][0] = samples[i*2];     // I (real)
        in[i][1] = samples[i*2 + 1]; // Q (imag)
    }
    
    // Выполняем FFT
    fftw_execute(plan);
    
    // Вычисляем мощность спектра в dB
    float max_val = 1e-10;
    for(int i = 0; i < n_fft/2; i++) {
        float mag = out[i][0]*out[i][0] + out[i][1]*out[i][1];
        if(mag > max_val) max_val = mag;
    }
    
    for(int i = 0; i < n_fft/2; i++) {
        float mag = out[i][0]*out[i][0] + out[i][1]*out[i][1];
        spectrum_out[i] = 10 * log10(mag / max_val + 1e-10); // Нормализованный спектр в dB
    }
    
    pthread_mutex_unlock(&fft_mutex);
}

void *sdr_rx_thread(void *arg) {
    struct SharedData *shared = (struct SharedData*)arg;

    SoapySDRKwargs args = {};
    SoapySDRKwargs_set(&args, "driver", "plutosdr");
    SoapySDRKwargs_set(&args, "uri", shared->sdr_uri);
    
    printf("RX: Opening SDR at %s...\n", shared->sdr_uri);
    SoapySDRDevice *sdr = SoapySDRDevice_make(&args);
    SoapySDRKwargs_clear(&args);

    if (!sdr) {
        printf("RX: ERROR - Failed to open SDR\n");
        return NULL;
    }

    printf("RX: SDR opened successfully\n");

    int sample_rate = 1000000; // 1 MSPS
    double rx_freq = 800e6; // 800 MHz

    SoapySDRDevice_setSampleRate(sdr, SOAPY_SDR_RX, 0, sample_rate);
    SoapySDRDevice_setFrequency(sdr, SOAPY_SDR_RX, 0, rx_freq, NULL);
    SoapySDRDevice_setGain(sdr, SOAPY_SDR_RX, 0, 40.0);

    size_t channels[] = {0};
    const size_t channel_count = 1;
    
    printf("RX: Setting up stream...\n");
    SoapySDRStream *rxStream = SoapySDRDevice_setupStream(sdr, SOAPY_SDR_RX, SOAPY_SDR_CS16, channels, channel_count, NULL);
    if (!rxStream) {
        printf("RX: Failed to setup stream\n");
        SoapySDRDevice_unmake(sdr);
        return NULL;
    }

    SoapySDRDevice_activateStream(sdr, rxStream, 0, 0, 0);

    size_t rx_mtu = SoapySDRDevice_getStreamMTU(sdr, rxStream);
    int16_t *rx_buffer = (int16_t*)malloc(2 * rx_mtu * sizeof(int16_t));

    const long timeoutUs = 400000;
    
    int Nsps = 10;
    float p2 = 0;
    float zeta = 0.707;
    float BnTs = 0.005;
    float theta = (BnTs / Nsps) / (zeta + 1/(4*zeta));
    float K1 = (4*zeta*theta) / (1 + 2*zeta*theta + theta*theta);
    float K2 = (4*theta*theta) / (1 + 2*zeta*theta + theta*theta);
    int no_signal_counter = 0;
    
    // Буферы для спектра
    float temp_spectrum[1024];
    float power_buffer[1000] = {0};
    int power_idx = 0;
    
    printf("RX: Starting receive loop. MTU = %zu samples\n", rx_mtu);
    int cycle_count = 0;
    int fft_counter = 0;

    while (shared->running && global_running) {
        void *rx_buffs[] = {rx_buffer};
        int flags;
        long long timeNs;
        
        int sr = SoapySDRDevice_readStream(sdr, rxStream, rx_buffs, rx_mtu, &flags, &timeNs, timeoutUs);

    if (sr > 0) {
        cycle_count++;
        
        // Вычисляем мощность сигнала СРАЗУ
        float instant_power = 0;
        float max_val = 0;
        for (int i = 0; i < sr; i++) {
            float i_val = rx_buffer[i * 2] / 32768.0f;
            float q_val = rx_buffer[i * 2 + 1] / 32768.0f;
            instant_power += i_val*i_val + q_val*q_val;
            if(fabs(i_val) > max_val) max_val = fabs(i_val);
            if(fabs(q_val) > max_val) max_val = fabs(q_val);
            
            ring_buffer_write(&shared->raw_rb, i_val);
            ring_buffer_write(&shared->raw_rb, q_val);
        }
        instant_power /= sr;
        
        // Обновляем мощность
        power_buffer[power_idx % 1000] = instant_power;
        power_idx++;
        if(power_idx > 1000) {
            float avg_power = 0;
            for(int i = 0; i < 1000; i++) avg_power += power_buffer[i];
            avg_power /= 1000;
            shared->current_signal_power = avg_power;
        }
        
        // Печатаем отладочную информацию
        if(cycle_count % 50 == 0) {
            printf("RX: Instant power=%.6f, Max=%.3f, Avg=%.6f\n", 
                instant_power, max_val, shared->current_signal_power);
        }
            
            apply_matched_filter_with_downsampling(&shared->raw_rb, &shared->filtered_i_rb, 
                                                  &shared->filtered_q_rb, Nsps);
            
            // Вычисляем спектр каждые 10 циклов, даже если сигнал слабый
            fft_counter++;
            if(fft_counter % 10 == 0) {
                int avail = ring_buffer_available(&shared->raw_rb);
                if(avail >= 2048 * 2) {
                    float temp_buf[2048 * 2];
                    ring_buffer_peek(&shared->raw_rb, temp_buf, 0, 2048 * 2);
                    
                    // Нормализуем перед FFT
                    float max_val = 1e-10;
                    for(int i = 0; i < 2048 * 2; i++) {
                        if(fabs(temp_buf[i]) > max_val) max_val = fabs(temp_buf[i]);
                    }
                    
                    if(max_val > 0.01) { // Если есть сигнал
                        for(int i = 0; i < 2048 * 2; i++) {
                            temp_buf[i] /= max_val; // Нормализация
                        }
                    }
                    
                    compute_spectrum(temp_buf, 2048, temp_spectrum, 1024);
                    
                    pthread_mutex_lock(&fft_mutex);
                    memcpy(shared->last_spectrum, temp_spectrum, 1024 * sizeof(float));
                    shared->spectrum_available = 1;
                    pthread_mutex_unlock(&fft_mutex);
                    
                    if(cycle_count % 50 == 0) {
                        printf("RX: Spectrum updated, max_val=%.3f\n", max_val);
                    }
                }
            }
            
            // Обработка отфильтрованных символов
            int available_filtered = ring_buffer_available(&shared->filtered_i_rb);
            const int SYMBOLS_TO_PROCESS = 200;
            int symbols_processed = 0;
            
            while (available_filtered > 0 && symbols_processed < SYMBOLS_TO_PROCESS && 
                   shared->running && global_running) {
                
                int read_pos = shared->filtered_i_rb.read_pos;
                
                if (read_pos != shared->filtered_i_rb.write_pos) {
                    float filtered_i = shared->filtered_i_rb.buffer[read_pos];
                    float filtered_q = shared->filtered_q_rb.buffer[read_pos];
                    
                    float amplitude = fabs(filtered_i) + fabs(filtered_q);
                    
                    if (amplitude > 0.05) {
                        float e = gardner_ted(&shared->filtered_i_rb, &shared->filtered_q_rb, 0, Nsps);
                        
                        float p1 = e * K1;
                        p2 = p2 + p1 + e * K2;
                        
                        if (p2 > 0.5f) p2 -= 1.0f;
                        if (p2 < -0.5f) p2 += 1.0f;
                        
                        no_signal_counter = 0;
                    } else {
                        no_signal_counter++;
                        if (no_signal_counter > 200) {
                            p2 = 0;
                            no_signal_counter = 0;
                        }
                    }
                    
                    ring_buffer_advance(&shared->filtered_i_rb, 1);
                    ring_buffer_advance(&shared->filtered_q_rb, 1);
                    symbols_processed++;
                    available_filtered--;
                } else {
                    break;
                }
            }
            
            if (cycle_count % 100 == 0) {
                printf("RX: Power=%.3f, Filtered=%d, FFT=%s\n", 
                       shared->current_signal_power, 
                       ring_buffer_available(&shared->filtered_i_rb),
                       shared->spectrum_available ? "available" : "waiting");
            }
        } else if (sr == SOAPY_SDR_TIMEOUT) {
            // Таймаут - нормально
        } else {
            printf("RX: Read stream error: %d\n", sr);
        }
    }

    printf("RX: Cleaning up...\n");
    free(rx_buffer);

    SoapySDRDevice_deactivateStream(sdr, rxStream, 0, 0);
    SoapySDRDevice_closeStream(sdr, rxStream);
    SoapySDRDevice_unmake(sdr);
    
    printf("RX: Thread exiting\n");
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <sdr_uri>\n", argv[0]);
        printf("Example: %s usb:1.5.5\n", argv[0]);
        printf("Example: %s ip:192.168.3.1\n", argv[0]);
        return EXIT_FAILURE;
    }

    signal(SIGINT, sigint_handler);

    struct SharedData shared;
    shared.sdr_uri = argv[1];
    shared.current_freq_offset = 0;
    shared.current_signal_power = 0;
    shared.spectrum_available = 0;
    memset(shared.last_spectrum, 0, sizeof(shared.last_spectrum));
    
    printf("BPSK Receiver with Spectrum\n");
    printf("SDR URI: %s\n", shared.sdr_uri);

    init_ring_buffer(&shared.raw_rb, 400000);
    init_ring_buffer(&shared.filtered_i_rb, 100000);
    init_ring_buffer(&shared.filtered_q_rb, 100000);
    
    shared.running = 1;

    // Запуск SDL и ImGui
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }
    
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    
    SDL_Window* window = SDL_CreateWindow(
        "BPSK Receiver with Spectrum", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 720, 
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    
    if (!window) {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }
    
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        printf("SDL_GL_CreateContext Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        printf("GLEW Error: %s\n", glewGetErrorString(err));
        SDL_GL_DeleteContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init("#version 330");

    const int DISPLAY_SIZE = 20000;
    float *i_display = (float*)malloc(DISPLAY_SIZE * sizeof(float));
    float *q_display = (float*)malloc(DISPLAY_SIZE * sizeof(float));
    float *filtered_i_display = (float*)malloc(DISPLAY_SIZE * sizeof(float));
    float *filtered_q_display = (float*)malloc(DISPLAY_SIZE * sizeof(float));
    
    static FILE* filtered_file = NULL;
    static int file_initialized = 0;
    static long long symbol_counter = 0;
    
    int available_samples = 0;
    int samples_to_show = 0;
    int available_filtered = 0;
    int filtered_to_show = 0;
    
    static int raw_points_to_show = 5000;
    static int filtered_points_to_show = 2000;
    static int fft_size = 1024;
    static float display_spectrum[1024] = {0};
    static float freq_bins[1024];

    // Инициализация частотных бинов для отображения
    for(int i = 0; i < fft_size; i++) {
        freq_bins[i] = (float)i;
    }

    // Запуск SDR потока
    pthread_t rx_thread;
    int thread_err = pthread_create(&rx_thread, NULL, sdr_rx_thread, (void*)&shared);
    if (thread_err != 0) {
        printf("Failed to create RX thread: %d\n", thread_err);
        return EXIT_FAILURE;
    }

    printf("GUI: Starting main loop\n");
    int frame_count = 0;

    while (global_running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                global_running = 0;
                shared.running = 0;
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_None);

        frame_count++;

        // Сбор данных для отображения
        available_samples = ring_buffer_available(&shared.raw_rb) / 2;
        samples_to_show = (available_samples < raw_points_to_show) ? available_samples : raw_points_to_show;
        
        if (samples_to_show > 0 && samples_to_show <= DISPLAY_SIZE) {
            ring_buffer_peek(&shared.raw_rb, i_display, 0, samples_to_show * 2);
            for(int i = 0; i < samples_to_show; i++) {
                float tmp_i = i_display[i*2];
                float tmp_q = i_display[i*2 + 1];
                i_display[i] = tmp_i;
                q_display[i] = tmp_q;
            }
        }
        
        available_filtered = ring_buffer_available(&shared.filtered_i_rb);
        filtered_to_show = (available_filtered < filtered_points_to_show) ? available_filtered : filtered_points_to_show;
        
        if (filtered_to_show > 0 && filtered_to_show <= DISPLAY_SIZE) {
            ring_buffer_peek(&shared.filtered_i_rb, filtered_i_display, 0, filtered_to_show);
            ring_buffer_peek(&shared.filtered_q_rb, filtered_q_display, 0, filtered_to_show);
            
            static struct CostasLoop costas;
            static int costas_initialized = 0;

            if (!costas_initialized) {
                costas.phase = 0.0f;
                costas.freq = 0.0f;
                costas.alpha = 0.05f;
                costas.beta = 0.001f;
                costas_initialized = 1;
            }

            for (int i = 0; i < filtered_to_show; i++) {
                costas_loop_update(&costas, &filtered_i_display[i], &filtered_q_display[i]);
            }
            shared.current_freq_offset = costas.freq;
        }

        // Получаем последний спектр
        if(shared.spectrum_available) {
            pthread_mutex_lock(&fft_mutex);
            memcpy(display_spectrum, shared.last_spectrum, fft_size * sizeof(float));
            pthread_mutex_unlock(&fft_mutex);
            
            if(frame_count % 60 == 0) {
                printf("GUI: Spectrum updated, first value=%.2f\n", display_spectrum[0]);
            }
        }

        // GUI элементы
        ImGui::Begin("Receiver Control");
        ImGui::Text("SDR URI: %s", shared.sdr_uri);
        ImGui::Text("Signal Power: %.6f", shared.current_signal_power);
        ImGui::Text("Freq Offset: %.6f", shared.current_freq_offset);
        ImGui::Text("Spectrum: %s", shared.spectrum_available ? "Available" : "Not ready");
        ImGui::Separator();
        ImGui::Text("Raw buffer: %d/%d samples", available_samples, shared.raw_rb.size/2);
        ImGui::Text("Filtered buffer: %d/%d symbols", available_filtered, shared.filtered_i_rb.size);
        ImGui::Separator();
        
        ImGui::Text("Raw IQ:");
        ImGui::SliderInt("Points to show", &raw_points_to_show, 100, 20000);
        ImGui::Text("Displayed: %d", samples_to_show);
        ImGui::Separator();
        
        ImGui::Text("Filtered BPSK:");
        ImGui::SliderInt("Symbols to show", &filtered_points_to_show, 100, 10000);
        ImGui::Text("Displayed: %d", filtered_to_show);
        ImGui::Text("Symbols saved: %lld", symbol_counter);
        ImGui::Separator();
        
        ImGui::Text("Status: %s", shared.running ? "Running" : "Stopped");
        ImGui::End();

        ImGui::Begin("Raw IQ Constellation");
        if (ImPlot::BeginPlot("Raw IQ", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("I", "Q");
            ImPlot::SetupAxesLimits(-1.2, 1.2, -1.2, 1.2, ImPlotCond_Once);
            
            if (samples_to_show > 0) {
                ImPlot::PlotScatter("Raw", i_display, q_display, samples_to_show);
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Filtered BPSK Constellation");
        if (ImPlot::BeginPlot("Filtered BPSK", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("I", "Q");
            ImPlot::SetupAxesLimits(-1.2, 1.2, -1.2, 1.2, ImPlotCond_Once);
            
            if (filtered_to_show > 0) {
                ImPlot::PlotScatter("Filtered", filtered_i_display, filtered_q_display, filtered_to_show);
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Signal Spectrum");
        if (ImPlot::BeginPlot("Spectrum", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("Frequency Bin", "Magnitude (dB)");
            ImPlot::SetupAxesLimits(0, fft_size-1, -60, 10, ImPlotCond_Once);
            
            if (shared.spectrum_available) {
                ImPlot::PlotLine("Spectrum", freq_bins, display_spectrum, fft_size);
            } else {
                // Показываем тестовые данные, если спектра нет
                float test_data[1024];
                for(int i = 0; i < fft_size; i++) {
                    test_data[i] = -40 + 10 * sinf(i * 0.1f);
                }
                ImPlot::PlotLine("No Signal", freq_bins, test_data, fft_size);
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Data Recording");
        if(ImGui::Button("Save symbols to file")) {
            if(!file_initialized) {
                filtered_file = fopen("symb_after_rx.pcm", "wb");
                if(filtered_file) {
                    file_initialized = 1;
                    printf("Started recording symbols\n");
                }
            } else {
                fclose(filtered_file);
                file_initialized = 0;
                printf("Stopped recording, total symbols: %lld\n", symbol_counter);
            }
        }
        
        if(file_initialized && filtered_to_show > 0) {
            for(int i = 0; i < filtered_to_show; i++) {
                int16_t i_int = (int16_t)(filtered_i_display[i] * 32767);
                int16_t q_int = (int16_t)(filtered_q_display[i] * 32767);
                fwrite(&i_int, sizeof(int16_t), 1, filtered_file);
                fwrite(&q_int, sizeof(int16_t), 1, filtered_file);
                symbol_counter++;
            }
        }
        ImGui::End();

        ImGui::Render();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);
        SDL_Delay(16);
    }

    shared.running = 0;
    pthread_join(rx_thread, NULL);

    if (filtered_file) {
        fclose(filtered_file);
        printf("Closed symb_after_rx.pcm, total symbols: %lld\n", symbol_counter);
    }

    free(i_display);
    free(q_display);
    free(filtered_i_display);
    free(filtered_q_display);
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    free(shared.raw_rb.buffer);
    free(shared.filtered_i_rb.buffer);
    free(shared.filtered_q_rb.buffer);

    // Очистка FFTW
    fftw_cleanup();

    printf("Receiver shutdown complete\n");
    return EXIT_SUCCESS;
}