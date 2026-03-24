#include <GL/glew.h>
#include <SDL2/SDL.h>

#include <SoapySDR/Device.h> 
#include <SoapySDR/Formats.h>
#include <stdio.h> 
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <complex.h>
#include <fcntl.h>
#include <sys/types.h>
#include <string.h>
#include <math.h>

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

struct SharedData {
    char *sdr_uri;
    
    struct LockFreeRingBuffer raw_rb;
    struct LockFreeRingBuffer filtered_i_rb;
    struct LockFreeRingBuffer filtered_q_rb;

    pthread_mutex_t mutex;
    
    int running;
};

void init_ring_buffer(struct LockFreeRingBuffer *rb, int size) {
    rb->buffer = (float*)malloc(size * sizeof(float));
    memset(rb->buffer, 0, size * sizeof(float));
    rb->size = size;
    rb->write_pos = 0;
    rb->read_pos = 0;
}

int ring_buffer_write(struct LockFreeRingBuffer *rb, float value) {
    int write_pos = rb->write_pos;
    int read_pos = rb->read_pos;
    
    int next_write_pos = (write_pos + 1) % rb->size;
    
    if (next_write_pos == read_pos) {
        return 0;
    }
    
    rb->buffer[write_pos] = value;
    rb->write_pos = next_write_pos;
    
    return 1;
}

int ring_buffer_read(struct LockFreeRingBuffer *rb, float *value) {
    int read_pos = rb->read_pos;
    int write_pos = rb->write_pos;
    
    if (read_pos == write_pos) {
        return 0;
    }
    
    *value = rb->buffer[read_pos];
    rb->read_pos = (read_pos + 1) % rb->size;
    
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

void ring_buffer_peek(struct LockFreeRingBuffer *rb, float *dest, int start_offset, int count) {
    int read_pos = rb->read_pos;
    
    for (int i = 0; i < count; i++) {
        int pos = (read_pos + start_offset + i) % rb->size;
        dest[i] = rb->buffer[pos];
    }
}

void ring_buffer_advance(struct LockFreeRingBuffer *rb, int count) {
    int read_pos = rb->read_pos;
    rb->read_pos = (read_pos + count) % rb->size;
}

int *to_bpsk(int *bit_arr, int length) {
    int *bpsk_arr = (int *)malloc(length * sizeof(int));
    for(int i = 0; i < length; i++) {
        if(bit_arr[i] == 0) {
            bpsk_arr[i] = 1;
        } else {
            bpsk_arr[i] = -1;
        }
    }
    return bpsk_arr;
}

int *upsampling(int *bpsk_arr, int length) {
    int count = 0;
    int *bpsk_after_upsampling = (int *)malloc(length * 10 * sizeof(int));
    
    for(int i = 0; i < length; i++) {
        if (i > 0) {
            for(int j = 0; j < 9; j++) {
                bpsk_after_upsampling[count] = 0;
                count++;
            }
        }
        bpsk_after_upsampling[count] = bpsk_arr[i];
        count++;
    }
    return bpsk_after_upsampling;
}

int *convolution(int *upsampling_arr, int *impulse_arr, int length, int impulse_length) {
    int result_length = length;
    int *upsampl_after_conv = (int *)malloc(result_length * sizeof(int));
    
    for (int i = 0; i < result_length; i++) {
        upsampl_after_conv[i] = 0;
    }
    
    for (int n = 0; n < result_length; n++) {
        for (int k = 0; k < impulse_length; k++) {
            if (n - k >= 0 && n - k < length) {
                upsampl_after_conv[n] += upsampling_arr[n - k] * impulse_arr[k];
            }
        }
    }
    
    return upsampl_after_conv;
}

float gardner_ted(struct LockFreeRingBuffer *filtered_i_rb, struct LockFreeRingBuffer *filtered_q_rb, int symbol_idx, int Nsps) {
    (void)Nsps;
    
    float i_early, i_middle, i_late, q_early, q_middle, q_late;
    int read_pos = filtered_i_rb->read_pos;
    
    int early = (read_pos + symbol_idx - 1 + filtered_i_rb->size) % filtered_i_rb->size;
    int middle = (read_pos + symbol_idx) % filtered_i_rb->size;
    int late = (read_pos + symbol_idx + 1) % filtered_i_rb->size;
    
    i_early = filtered_i_rb->buffer[early];
    i_middle = filtered_i_rb->buffer[middle];
    i_late = filtered_i_rb->buffer[late];
    
    q_early = filtered_q_rb->buffer[early];
    q_middle = filtered_q_rb->buffer[middle];
    q_late = filtered_q_rb->buffer[late];
    
    float error = ((i_late - i_early) * i_middle) + ((q_late - q_early) * q_middle);
    
    return error;
}

void apply_matched_filter_with_downsampling(struct LockFreeRingBuffer *raw_rb, 
                                           struct LockFreeRingBuffer *filtered_i_rb,
                                           struct LockFreeRingBuffer *filtered_q_rb, int Nsps) {
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
            if (sample_pos >= 0 && sample_pos < raw_rb->size) {
                conv_i += raw_rb->buffer[sample_pos] * h[k];
            }
        }
        
        for (int k = 0; k < Nsps; k++) {
            int sample_pos = (pos + 1 - k * 2 + raw_rb->size) % raw_rb->size;
            if (sample_pos >= 0 && sample_pos < raw_rb->size) {
                conv_q += raw_rb->buffer[sample_pos] * h[k];
            }
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

void *sdr_thread(void *arg) {
    struct SharedData *shared = (struct SharedData*)arg;

    int base_bit_arr[] = {0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,1,0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,1,0,0,0,1,0};
    int base_len = sizeof(base_bit_arr) / sizeof(base_bit_arr[0]);
    
    int len_arr = base_len * 100;
    int *bit_arr = (int*)malloc(len_arr * sizeof(int));
    
    for(int rep = 0; rep < 100; rep++) {
        for(int i = 0; i < base_len; i++) {
            bit_arr[rep * base_len + i] = base_bit_arr[i];
        }
    }
    
    int *bpsk_arr = to_bpsk(bit_arr, len_arr);
    int *bpsk_after_arr = upsampling(bpsk_arr, len_arr);

    int pulse[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int pulse_length = 10;

    int *conv_result = convolution(bpsk_after_arr, pulse, len_arr * 10, pulse_length);
    int conv_length = len_arr * 10;

    int repeat_count = 5;
    int total_samples = conv_length * repeat_count;
    int16_t *tx_samples = (int16_t *)malloc(total_samples * 2 * sizeof(int16_t));
    int scale_factor = 3000;
    
    for(int rep = 0; rep < repeat_count; rep++) {
        for(int i = 0; i < conv_length; i++) {
            int idx = rep * conv_length + i;

            static float phase = 0.0f;
            // частотный оффсет (нормированный, относительно Fs)
            float freq_offset = 0.0;   // пробуй 0.005–0.02

            float s = (float)conv_result[i];

            // генерация комплексной несущей
            float cos_p = cosf(phase);
            float sin_p = sinf(phase);

            // BPSK на несущей
            float i_val = s * cos_p;
            float q_val = s * sin_p;

            // масштабирование
            tx_samples[idx * 2]     = (int16_t)(i_val * scale_factor);
            tx_samples[idx * 2 + 1] = (int16_t)(q_val * scale_factor);

            // обновляем фазу (CFO)
            phase += freq_offset;

            // нормализация
            if (phase > 2 * M_PI) phase -= 2 * M_PI;
        }
    }

    printf("Total bits: %d\n", len_arr);
    printf("Total samples after upsampling and convolution: %d\n", conv_length);
    printf("Total transmitted samples (with repeats): %d\n", total_samples);

    SoapySDRKwargs args = {};
    SoapySDRKwargs_set(&args, "driver", "plutosdr"); 
    SoapySDRKwargs_set(&args, "uri", shared->sdr_uri);
    SoapySDRKwargs_set(&args, "direct", "1");
    
    SoapySDRDevice *sdr = SoapySDRDevice_make(&args);
    SoapySDRKwargs_clear(&args);

    int sample_rate = 1e6;
    double tx_freq = 800e6;
    double rx_freq = 800e6 + 50e3; // +50 kHz оффсет

    SoapySDRDevice_setSampleRate(sdr, SOAPY_SDR_TX, 0, sample_rate);
    SoapySDRDevice_setFrequency(sdr, SOAPY_SDR_TX, 0, tx_freq, NULL);
    SoapySDRDevice_setSampleRate(sdr, SOAPY_SDR_RX, 0, sample_rate);
    SoapySDRDevice_setFrequency(sdr, SOAPY_SDR_RX, 0, rx_freq, NULL);

    size_t channels[] = {0};
    const size_t channel_count = 1;
    
    SoapySDRDevice_setGain(sdr, SOAPY_SDR_TX, 0, 80.0);
    SoapySDRDevice_setGain(sdr, SOAPY_SDR_RX, 0, 20.0);

    SoapySDRStream *txStream = SoapySDRDevice_setupStream(sdr, SOAPY_SDR_TX, SOAPY_SDR_CS16, channels, channel_count, NULL);
    SoapySDRStream *rxStream = SoapySDRDevice_setupStream(sdr, SOAPY_SDR_RX, SOAPY_SDR_CS16, channels, channel_count, NULL);

    SoapySDRDevice_activateStream(sdr, txStream, 0, 0, 0);
    SoapySDRDevice_activateStream(sdr, rxStream, 0, 0, 0);

    size_t tx_mtu = SoapySDRDevice_getStreamMTU(sdr, txStream);
    size_t rx_mtu = SoapySDRDevice_getStreamMTU(sdr, rxStream);

    int16_t *tx_buff = (int16_t*)malloc(2 * tx_mtu * sizeof(int16_t));
    int16_t *rx_buffer = (int16_t*)malloc(2 * rx_mtu * sizeof(int16_t));

    const long timeoutUs = 400000;

    void *rx_buffs[] = {rx_buffer};
    int rx_flags;
    long long timeNs;
    int sr = SoapySDRDevice_readStream(sdr, rxStream, rx_buffs, rx_mtu, &rx_flags, &timeNs, timeoutUs);
    
    long long tx_time = 0;
    if (sr > 0) {
        tx_time = timeNs + (5 * 1000 * 1000);
    }
    
    int Nsps = 10;
    float p2 = 0;
    float zeta = 0.707;
    float BnTs = 0.005;
    float theta = (BnTs / Nsps) / (zeta + 1/(4*zeta));
    float K1 = (4*zeta*theta) / (1 + 2*zeta*theta + theta*theta);
    float K2 = (4*theta*theta) / (1 + 2*zeta*theta + theta*theta);
    int no_signal_counter = 0;
    
    while (shared->running) {
        int total_samples_sent = 0;
        int flags = SOAPY_SDR_HAS_TIME;

        while (total_samples_sent < total_samples && shared->running) {
            int samples_to_send;
            if (total_samples - total_samples_sent < (int)tx_mtu) {
                samples_to_send = total_samples - total_samples_sent;
            } else {
                samples_to_send = (int)tx_mtu;
            }

            for (int i = 0; i < samples_to_send * 2; i++) {
                tx_buff[i] = tx_samples[total_samples_sent * 2 + i] * 1500;
            }

            for (int i = samples_to_send * 2; i < (int)tx_mtu * 2; i++) {
                tx_buff[i] = 0;
            }

            void *tx_buffs[] = {tx_buff};
            SoapySDRDevice_writeStream(sdr, txStream, (const void * const*)tx_buffs, tx_mtu, &flags, tx_time, timeoutUs);
            
            total_samples_sent += samples_to_send;
            tx_time += (samples_to_send * 1000000000LL) / sample_rate;
            flags = SOAPY_SDR_HAS_TIME;
        }
        
        tx_time += (100 * 1000000LL);

        int total_samples_received = 0;
        while (total_samples_received < total_samples * 2 && shared->running) {
            void *rx_buffs[] = {rx_buffer};
            int flags;
            long long timeNs;
            
            int sr = SoapySDRDevice_readStream(sdr, rxStream, rx_buffs, rx_mtu, &flags, &timeNs, timeoutUs);

            if (sr > 0) {
                for (int i = 0; i < sr; i++) {
                    ring_buffer_write(&shared->raw_rb, rx_buffer[i * 2] / 32768.0f);
                    ring_buffer_write(&shared->raw_rb, rx_buffer[i * 2 + 1] / 32768.0f);
                }
                
                apply_matched_filter_with_downsampling(&shared->raw_rb, &shared->filtered_i_rb, &shared->filtered_q_rb, Nsps);
                
                int available_filtered = ring_buffer_available(&shared->filtered_i_rb);
                
                const int SYMBOLS_TO_PROCESS = 200;
                int symbols_processed = 0;
                
                while (available_filtered > 0 && symbols_processed < SYMBOLS_TO_PROCESS) {
                    float filtered_i, filtered_q;
                    int read_pos = shared->filtered_i_rb.read_pos;
                    
                    if (read_pos != shared->filtered_i_rb.write_pos) {
                        filtered_i = shared->filtered_i_rb.buffer[read_pos];
                        filtered_q = shared->filtered_q_rb.buffer[read_pos];
                        
                        float amplitude = fabs(filtered_i) + fabs(filtered_q);
                        
                        if (amplitude > 0.01) {
                            float e = gardner_ted(&shared->filtered_i_rb, &shared->filtered_q_rb, 0, Nsps);
                            
                            float p1 = e * K1;
                            p2 = p2 + p1 + e * K2;
                            
                            if (p2 > 0.5f) p2 -= 1.0f;
                            if (p2 < -0.5f) p2 += 1.0f;
                            
                            no_signal_counter = 0;
                        } else {
                            no_signal_counter++;
                            if (no_signal_counter > 100) {
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
                
                total_samples_received += sr;
            }
        }
        
        printf("Transmission cycle completed. Total samples received: %d\n", total_samples_received);
    }

    free(tx_buff);
    free(rx_buffer);
    free(tx_samples);
    free(bpsk_arr);
    free(bpsk_after_arr);
    free(conv_result);
    free(bit_arr);

    SoapySDRDevice_deactivateStream(sdr, txStream, 0, 0);
    SoapySDRDevice_deactivateStream(sdr, rxStream, 0, 0);
    SoapySDRDevice_closeStream(sdr, txStream);
    SoapySDRDevice_closeStream(sdr, rxStream);
    SoapySDRDevice_unmake(sdr);
    
    return NULL;
}

struct CostasLoop {
    float phase;
    float freq;
    float alpha;
    float beta;
};

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

void *imgui_thread(void *arg) {
    struct SharedData *shared = (struct SharedData*)arg;
    
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER);
    SDL_Window* window = SDL_CreateWindow(
        "SDR IQ Viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);

    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;
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
    
    bool running = true;
    int frame_counter = 0;
    
    int available_samples = 0;
    int samples_to_show = 0;
    int available_filtered = 0;
    int filtered_to_show = 0;
    
    static int raw_points_to_show = 5000;
    static int filtered_points_to_show = 2000;
    
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                running = false;
                shared->running = 0;
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_None);

        frame_counter++;
        
        available_samples = ring_buffer_available(&shared->raw_rb) / 2;
        samples_to_show = (available_samples < raw_points_to_show) ? available_samples : raw_points_to_show;
        
        if (samples_to_show > 0 && samples_to_show <= DISPLAY_SIZE) {
            int read_pos = shared->raw_rb.read_pos;
            int write_pos = shared->raw_rb.write_pos;
            
            int start_pos;
            if (write_pos >= read_pos) {
                start_pos = write_pos - samples_to_show * 2;
                if (start_pos < 0) start_pos = 0;
            } else {
                start_pos = write_pos - samples_to_show * 2;
                if (start_pos < 0) start_pos = shared->raw_rb.size + start_pos;
            }
            start_pos = (start_pos / 2) * 2;
            
            for (int i = 0; i < samples_to_show; i++) {
                int pos = (start_pos + i * 2) % shared->raw_rb.size;
                i_display[i] = shared->raw_rb.buffer[pos];
                q_display[i] = shared->raw_rb.buffer[(pos + 1) % shared->raw_rb.size];
            }
        }
        
        available_filtered = ring_buffer_available(&shared->filtered_i_rb);
        filtered_to_show = (available_filtered < filtered_points_to_show) ? available_filtered : filtered_points_to_show;
        
        if (filtered_to_show > 0 && filtered_to_show <= DISPLAY_SIZE) {
            int read_pos = shared->filtered_i_rb.read_pos;
            int write_pos = shared->filtered_i_rb.write_pos;
            
            int start_pos;
            if (write_pos >= read_pos) {
                start_pos = write_pos - filtered_to_show;
                if (start_pos < 0) start_pos = 0;
            } else {
                start_pos = write_pos - filtered_to_show;
                if (start_pos < 0) start_pos = shared->filtered_i_rb.size + start_pos;
            }
            
            static struct CostasLoop costas = {0};
            static int costas_initialized = 0;

            if (!costas_initialized) {
                costas.phase = 0.0f;
                costas.freq = 0.0f;
                costas.alpha = 0.05f;
                costas.beta  = 0.0001f;
                costas_initialized = 1;
            }

            if (!file_initialized) {
                filtered_file = fopen("symb_after_rx.pcm", "wb");
                if (filtered_file) {
                    printf("Opened symb_after_rx.pcm for writing\n");
                    file_initialized = 1;
                }
            }

            if (file_initialized && filtered_file && filtered_to_show > 0) {
                for (int i = 0; i < filtered_to_show; i++) {
                    int pos = (start_pos + i) % shared->filtered_i_rb.size;

                    float i_val = shared->filtered_i_rb.buffer[pos];
                    float q_val = shared->filtered_q_rb.buffer[pos];

                    costas_loop_update(&costas, &i_val, &q_val);

                    filtered_i_display[i] = i_val;
                    filtered_q_display[i] = q_val;

                    int16_t i_int = (int16_t)(i_val * 32767);
                    int16_t q_int = (int16_t)(q_val * 32767);

                    fwrite(&i_int, sizeof(int16_t), 1, filtered_file);
                    fwrite(&q_int, sizeof(int16_t), 1, filtered_file);

                    symbol_counter++;
                }
            }
        }

        ImGui::Begin("SDR Control Panel");
        ImGui::Text("SDR URI: %s", shared->sdr_uri);
        ImGui::Text("Raw buffer size: %d samples", shared->raw_rb.size / 2);
        ImGui::Text("Filtered buffer size: %d symbols", shared->filtered_i_rb.size);
        ImGui::Separator();
        
        ImGui::Text("Raw IQ Settings:");
        ImGui::SliderInt("Raw points to show", &raw_points_to_show, 100, 20000);
        ImGui::Text("Available raw samples: %d", available_samples);
        ImGui::Text("Displayed raw samples: %d", samples_to_show);
        ImGui::Separator();
        
        ImGui::Text("Filtered BPSK Settings:");
        ImGui::SliderInt("Filtered points to show", &filtered_points_to_show, 100, 10000);
        ImGui::Text("Available filtered: %d", available_filtered);
        ImGui::Text("Displayed filtered: %d", filtered_to_show);
        ImGui::Separator();
        
        ImGui::Text("Application Status: %s", shared->running ? "Running" : "Stopped");
        ImGui::End();

        ImGui::Begin("Raw IQ Constellation");
        if (ImPlot::BeginPlot("Raw IQ Diagram", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("In-Phase (I)", "Quadrature (Q)");
            ImPlot::SetupAxesLimits(-1.2, 1.2, -1.2, 1.2, ImPlotCond_Once);
            
            if (samples_to_show > 0) {
                ImPlot::PlotScatter("Raw IQ", i_display, q_display, samples_to_show);
            }
            
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Filtered BPSK Constellation");
        if (ImPlot::BeginPlot("Filtered BPSK", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("In-Phase (I)", "Quadrature (Q)");
            ImPlot::SetupAxesLimits(-1.2, 1.2, -1.2, 1.2, ImPlotCond_Once);
            
            if (filtered_to_show > 0) {
                ImPlot::PlotScatter("Filtered BPSK", filtered_i_display, filtered_q_display, filtered_to_show);
            }
            
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Render();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);
    }

    if (filtered_file) {
        fclose(filtered_file);
        printf("Closed symb_after_rx.pcm, total symbols written: %lld\n", symbol_counter);
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
    
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <sdr_uri>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    struct SharedData shared;
    shared.sdr_uri = argv[1];
    
    init_ring_buffer(&shared.raw_rb, 400000); // 200000 I/Q pairs
    init_ring_buffer(&shared.filtered_i_rb, 100000);
    init_ring_buffer(&shared.filtered_q_rb, 100000);
    
    shared.running = 1;
    
    pthread_mutex_init(&shared.mutex, NULL);

    pthread_t pthreads[2];

    pthread_create(&pthreads[0], NULL, imgui_thread, (void*)&shared);
    pthread_create(&pthreads[1], NULL, sdr_thread, (void*)&shared);
    
    pthread_join(pthreads[0], NULL);
    pthread_join(pthreads[1], NULL);

    pthread_mutex_destroy(&shared.mutex);
    free(shared.raw_rb.buffer);
    free(shared.filtered_i_rb.buffer);
    free(shared.filtered_q_rb.buffer);

    return EXIT_SUCCESS;
}