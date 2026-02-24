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

struct SharedData {
    char *tx_uri;
    char *rx_uri;
    
    float *iq_buffer;     
    int buffer_size;        
    int buffer_write_pos;  
    int buffer_read_pos;    

    float *filtered_i_buffer;
    float *filtered_q_buffer;
    int filtered_buffer_size;
    int filtered_write_pos;
    int filtered_read_pos;

    pthread_mutex_t mutex;
    
    int running;
};

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

float gardner_ted(float *filtered_i, float *filtered_q, int symbol_idx, int Nsps, int buffer_size) {
    int early = (symbol_idx - 1 + buffer_size) % buffer_size;
    int middle = symbol_idx;
    int late = (symbol_idx + 1) % buffer_size;
    
    float i_early = filtered_i[early];
    float i_middle = filtered_i[middle];
    float i_late = filtered_i[late];
    
    float q_early = filtered_q[early];
    float q_middle = filtered_q[middle];
    float q_late = filtered_q[late];
    
    float error = ((i_late - i_early) * i_middle) + ((q_late - q_early) * q_middle);
    
    return error;
}

void apply_matched_filter_with_downsampling(float *iq_buffer, int read_pos, int write_pos, 
                                           int buffer_size, float *filtered_i, float *filtered_q,
                                           int filtered_size, int *filtered_write_pos, int Nsps) {
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
    
    int available = 0;
    if (write_pos >= read_pos) {
        available = write_pos - read_pos;
    } else {
        available = buffer_size - read_pos + write_pos;
    }
    
    if (available < Nsps) return;
    
    int start_pos = read_pos;
    int samples_to_process = available;
    
    for (int n = 0; n < samples_to_process; n++) {
        int pos = (start_pos + n) % buffer_size;
        float conv_i = 0;
        float conv_q = 0;
        
        for (int k = 0; k < Nsps; k++) {
            int sample_pos = (pos - k + buffer_size) % buffer_size;
            conv_i += iq_buffer[sample_pos * 2] * h[k];
            conv_q += iq_buffer[sample_pos * 2 + 1] * h[k];
        }
        
        sample_counter++;
        if (sample_counter >= Nsps) {
            filtered_i[*filtered_write_pos] = conv_i;
            filtered_q[*filtered_write_pos] = conv_q;
            *filtered_write_pos = (*filtered_write_pos + 1) % filtered_size;
            sample_counter = 0;
        }
    }
}

void *sdr_thread(void *arg) {
    struct SharedData *shared = (struct SharedData*)arg;

    int bit_arr[] = {0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,1,0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,1,0,0,0,1,0};
    int len_arr = sizeof(bit_arr) / sizeof(bit_arr[0]);
    
    int *bpsk_arr = to_bpsk(bit_arr, len_arr);
    int *bpsk_after_arr = upsampling(bpsk_arr, len_arr);

    int pulse[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int pulse_length = 10;

    int *conv_result = convolution(bpsk_after_arr, pulse, len_arr * 10, pulse_length);
    int conv_length = len_arr * 10;

    int repeat_count = 50;
    int total_samples = conv_length * repeat_count;
    int16_t *tx_samples = (int16_t *)malloc(total_samples * 2 * sizeof(int16_t));
    int scale_factor = 3000;
    
    for(int rep = 0; rep < repeat_count; rep++) {
        for(int i = 0; i < conv_length; i++) {
            int idx = rep * conv_length + i;
            tx_samples[idx * 2] = (int16_t)(conv_result[i] * scale_factor);
            tx_samples[idx * 2 + 1] = 0;
        }
    }

    SoapySDRKwargs tx_args = {};
    SoapySDRKwargs_set(&tx_args, "driver", "plutosdr"); 
    SoapySDRKwargs_set(&tx_args, "uri", shared->tx_uri);
    SoapySDRKwargs_set(&tx_args, "direct", "1");
    
    SoapySDRDevice *tx_sdr = SoapySDRDevice_make(&tx_args);
    SoapySDRKwargs_clear(&tx_args);

    SoapySDRKwargs rx_args = {};
    SoapySDRKwargs_set(&rx_args, "driver", "plutosdr"); 
    SoapySDRKwargs_set(&rx_args, "uri", shared->rx_uri);
    SoapySDRKwargs_set(&rx_args, "direct", "1");
    
    SoapySDRDevice *rx_sdr = SoapySDRDevice_make(&rx_args);
    SoapySDRKwargs_clear(&rx_args);

    int sample_rate = 1e6;
    int carrier_freq = 800e6;

    SoapySDRDevice_setSampleRate(tx_sdr, SOAPY_SDR_TX, 0, sample_rate);
    SoapySDRDevice_setFrequency(tx_sdr, SOAPY_SDR_TX, 0, carrier_freq, NULL);
    SoapySDRDevice_setSampleRate(rx_sdr, SOAPY_SDR_RX, 0, sample_rate);
    SoapySDRDevice_setFrequency(rx_sdr, SOAPY_SDR_RX, 0, carrier_freq, NULL);

    size_t channels[] = {0};
    const size_t channel_count = 1;
    
    SoapySDRDevice_setGain(tx_sdr, SOAPY_SDR_TX, 0, 80.0);
    SoapySDRDevice_setGain(rx_sdr, SOAPY_SDR_RX, 0, 20.0);

    SoapySDRStream *txStream = SoapySDRDevice_setupStream(tx_sdr, SOAPY_SDR_TX, SOAPY_SDR_CS16, channels, channel_count, NULL);
    SoapySDRStream *rxStream = SoapySDRDevice_setupStream(rx_sdr, SOAPY_SDR_RX, SOAPY_SDR_CS16, channels, channel_count, NULL);

    SoapySDRDevice_activateStream(tx_sdr, txStream, 0, 0, 0);
    SoapySDRDevice_activateStream(rx_sdr, rxStream, 0, 0, 0);

    size_t tx_mtu = SoapySDRDevice_getStreamMTU(tx_sdr, txStream);
    size_t rx_mtu = SoapySDRDevice_getStreamMTU(rx_sdr, rxStream);

    int16_t *tx_buff = (int16_t*)malloc(2 * tx_mtu * sizeof(int16_t));
    int16_t *rx_buffer = (int16_t*)malloc(2 * rx_mtu * sizeof(int16_t));

    const long timeoutUs = 400000;

    void *rx_buffs[] = {rx_buffer};
    int rx_flags;
    long long timeNs;
    int sr = SoapySDRDevice_readStream(rx_sdr, rxStream, rx_buffs, rx_mtu, &rx_flags, &timeNs, timeoutUs);
    
    long long tx_time = 0;
    if (sr > 0) {
        tx_time = timeNs + (5 * 1000 * 1000);
    }
    
    int Nsps = 10;
    float p2 = 0;
    float Kp = 0.01;
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
            int samples_to_send = (total_samples - total_samples_sent < tx_mtu) ? 
                                 (total_samples - total_samples_sent) : tx_mtu;

            for (int i = 0; i < samples_to_send * 2; i++) {
                tx_buff[i] = tx_samples[total_samples_sent * 2 + i] * 1500;
            }

            for (int i = samples_to_send * 2; i < tx_mtu * 2; i++) {
                tx_buff[i] = 0;
            }

            void *tx_buffs[] = {tx_buff};
            int st = SoapySDRDevice_writeStream(tx_sdr, txStream, (const void * const*)tx_buffs, tx_mtu, &flags, tx_time, timeoutUs);
            
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
            
            int sr = SoapySDRDevice_readStream(rx_sdr, rxStream, rx_buffs, rx_mtu, &flags, &timeNs, timeoutUs);

            if (sr > 0) {
                pthread_mutex_lock(&shared->mutex);
                
                for (int i = 0; i < sr * 2; i += 2) {
                    shared->iq_buffer[shared->buffer_write_pos * 2] = rx_buffer[i] / 32768.0f;
                    shared->iq_buffer[shared->buffer_write_pos * 2 + 1] = rx_buffer[i + 1] / 32768.0f;
                    
                    shared->buffer_write_pos = (shared->buffer_write_pos + 1) % shared->buffer_size;
                    
                    if (shared->buffer_write_pos == shared->buffer_read_pos) {
                        shared->buffer_read_pos = (shared->buffer_read_pos + 1) % shared->buffer_size;
                    }
                }
                
                apply_matched_filter_with_downsampling(shared->iq_buffer, shared->buffer_read_pos, shared->buffer_write_pos, 
                                   shared->buffer_size, shared->filtered_i_buffer, shared->filtered_q_buffer,
                                   shared->filtered_buffer_size, &shared->filtered_write_pos, Nsps);
                
                int available_filtered = 0;
                if (shared->filtered_write_pos >= shared->filtered_read_pos) {
                    available_filtered = shared->filtered_write_pos - shared->filtered_read_pos;
                } else {
                    available_filtered = shared->filtered_buffer_size - shared->filtered_read_pos + shared->filtered_write_pos;
                }
                
                const int SYMBOLS_TO_PROCESS = 5;
                int symbols_processed = 0;
                
                while (available_filtered > 0 && symbols_processed < SYMBOLS_TO_PROCESS) {
                    float amplitude = fabs(shared->filtered_i_buffer[shared->filtered_read_pos]) + 
                                     fabs(shared->filtered_q_buffer[shared->filtered_read_pos]);
                    
                    if (amplitude > 0.01) {
                        float e = gardner_ted(shared->filtered_i_buffer, shared->filtered_q_buffer, 
                                             shared->filtered_read_pos, Nsps, shared->filtered_buffer_size);
                        
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
                    
                    shared->filtered_read_pos = (shared->filtered_read_pos + 1) % shared->filtered_buffer_size;
                    symbols_processed++;
                    available_filtered--;
                }
                
                pthread_mutex_unlock(&shared->mutex);
                
                total_samples_received += sr;
            }
        }
    }

    free(tx_buff);
    free(rx_buffer);
    free(tx_samples);
    free(bpsk_arr);
    free(bpsk_after_arr);
    free(conv_result);

    SoapySDRDevice_deactivateStream(tx_sdr, txStream, 0, 0);
    SoapySDRDevice_deactivateStream(rx_sdr, rxStream, 0, 0);
    SoapySDRDevice_closeStream(tx_sdr, txStream);
    SoapySDRDevice_closeStream(rx_sdr, rxStream);
    SoapySDRDevice_unmake(tx_sdr);
    SoapySDRDevice_unmake(rx_sdr);
    
    return NULL;
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

    const int DISPLAY_SIZE = 2000;
    float *i_display = (float*)malloc(DISPLAY_SIZE * sizeof(float));
    float *q_display = (float*)malloc(DISPLAY_SIZE * sizeof(float));
    float *filtered_i_display = (float*)malloc(DISPLAY_SIZE * sizeof(float));
    float *filtered_q_display = (float*)malloc(DISPLAY_SIZE * sizeof(float));
    
    bool running = true;
    int frame_counter = 0;
    
    int available_samples = 0;
    int samples_to_show = 0;
    int available_filtered = 0;
    int filtered_to_show = 0;
    
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
        
        if (frame_counter % 3 == 0) {
            pthread_mutex_lock(&shared->mutex);
            
            if (shared->buffer_write_pos >= shared->buffer_read_pos) {
                available_samples = shared->buffer_write_pos - shared->buffer_read_pos;
            } else {
                available_samples = shared->buffer_size - shared->buffer_read_pos + shared->buffer_write_pos;
            }
            
            samples_to_show = (available_samples < DISPLAY_SIZE) ? available_samples : DISPLAY_SIZE;
            
            if (samples_to_show > 0) {
                int start_pos = (shared->buffer_write_pos - samples_to_show + shared->buffer_size) % shared->buffer_size;
                
                for (int i = 0; i < samples_to_show; i++) {
                    int pos = (start_pos + i) % shared->buffer_size;
                    i_display[i] = shared->iq_buffer[pos * 2];
                    q_display[i] = shared->iq_buffer[pos * 2 + 1];
                }
            }
            
            if (shared->filtered_write_pos >= shared->filtered_read_pos) {
                available_filtered = shared->filtered_write_pos - shared->filtered_read_pos;
            } else {
                available_filtered = shared->filtered_buffer_size - shared->filtered_read_pos + shared->filtered_write_pos;
            }
            
            filtered_to_show = (available_filtered < DISPLAY_SIZE) ? available_filtered : DISPLAY_SIZE;
            
            if (filtered_to_show > 0) {
                int start_pos = (shared->filtered_write_pos - filtered_to_show + shared->filtered_buffer_size) % shared->filtered_buffer_size;
                
                for (int i = 0; i < filtered_to_show; i++) {
                    int pos = (start_pos + i) % shared->filtered_buffer_size;
                    filtered_i_display[i] = shared->filtered_i_buffer[pos];
                    filtered_q_display[i] = shared->filtered_q_buffer[pos];
                }
            }
            
            pthread_mutex_unlock(&shared->mutex);
        }

        ImGui::Begin("SDR Control Panel");
        ImGui::Text("TX URI: %s", shared->tx_uri);
        ImGui::Text("RX URI: %s", shared->rx_uri);
        ImGui::Text("Buffer size: %d samples", shared->buffer_size);
        ImGui::Text("Available samples: %d", available_samples);
        ImGui::Text("Displayed samples: %d", samples_to_show);
        ImGui::Text("Available filtered: %d", available_filtered);
        ImGui::Separator();
        ImGui::Text("Application Status: %s", shared->running ? "Running" : "Stopped");
        ImGui::End();

        ImGui::Begin("Raw IQ Constellation");
        if (ImPlot::BeginPlot("Raw IQ Diagram", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("In-Phase (I)", "Quadrature (Q)");
            ImPlot::SetupAxesLimits(-1.2, 1.2, -1.2, 1.2);
            
            if (samples_to_show > 0) {
                ImPlot::PlotScatter("Raw IQ", i_display, q_display, samples_to_show);
            }
            
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Filtered BPSK Constellation");
        if (ImPlot::BeginPlot("Filtered BPSK", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("In-Phase (I)", "Quadrature (Q)");
            ImPlot::SetupAxesLimits(-1.2, 1.2, -1.2, 1.2);
            
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
    if (argc < 3) {
        printf("Usage: %s <tx_uri> <rx_uri>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    struct SharedData shared;
    shared.tx_uri = argv[1];
    shared.rx_uri = argv[2];
    shared.buffer_size = 20000;
    shared.iq_buffer = (float*)malloc(shared.buffer_size * 2 * sizeof(float));
    memset(shared.iq_buffer, 0, shared.buffer_size * 2 * sizeof(float));
    shared.buffer_write_pos = 0;
    shared.buffer_read_pos = 0;
    
    shared.filtered_buffer_size = 10000;
    shared.filtered_i_buffer = (float*)malloc(shared.filtered_buffer_size * sizeof(float));
    shared.filtered_q_buffer = (float*)malloc(shared.filtered_buffer_size * sizeof(float));
    memset(shared.filtered_i_buffer, 0, shared.filtered_buffer_size * sizeof(float));
    memset(shared.filtered_q_buffer, 0, shared.filtered_buffer_size * sizeof(float));
    shared.filtered_write_pos = 0;
    shared.filtered_read_pos = 0;
    
    shared.running = 1;
    
    pthread_mutex_init(&shared.mutex, NULL);

    pthread_t pthreads[2];

    pthread_create(&pthreads[0], NULL, imgui_thread, (void*)&shared);
    pthread_create(&pthreads[1], NULL, sdr_thread, (void*)&shared);
    
    pthread_join(pthreads[0], NULL);
    pthread_join(pthreads[1], NULL);

    pthread_mutex_destroy(&shared.mutex);
    free(shared.iq_buffer);
    free(shared.filtered_i_buffer);
    free(shared.filtered_q_buffer);

    return EXIT_SUCCESS;
}