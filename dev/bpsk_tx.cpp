#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>

#include <SoapySDR/Device.h>
#include <SoapySDR/Formats.h>
#include <SoapySDR/Errors.h>

int running = 1;
int16_t *tx_samples = NULL;
int total_samples = 0;
int sample_rate = 1000000; // 1 MSPS

// Структура пакета
struct PacketConfig {
    int preamble_len;
    int data_len;
    int total_len;
};

PacketConfig packet_cfg;

void sigint_handler(int sig) {
    (void)sig;
    printf("\nCaught SIGINT, shutting down...\n");
    running = 0;
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

// Генерация преамбулы для оценки частоты
void generate_preamble(int16_t *buffer, int &offset, int length) {
    printf("TX: Generating preamble (%d samples)...\n", length);
    
    // Преамбула: повторяющийся паттерн для оценки частоты
    // Используем последовательность с известной структурой
    for(int i = 0; i < length; i++) {
        // Модулирующий тон для оценки частоты
        float freq = 50000.0; // 50 kHz тон
        float phase = 2.0f * M_PI * freq * i / sample_rate;
        
        float i_val = cosf(phase);
        float q_val = sinf(phase);
        
        buffer[offset++] = (int16_t)(i_val * 3000);
        buffer[offset++] = (int16_t)(q_val * 3000);
    }
    
    // Добавляем уникальный маркер для синхронизации (короткая последовательность)
    for(int i = 0; i < 100; i++) {
        buffer[offset++] = 0;
        buffer[offset++] = 3000;
    }
    
    packet_cfg.preamble_len = length + 100;
}

// Функция для генерации тестовых данных
void generate_test_signal() {
    int base_bit_arr[] = {0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,1,0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,1,0,0,0,1,0};
    int base_len = sizeof(base_bit_arr) / sizeof(base_bit_arr[0]);
    
    // Увеличим количество повторений для более длительной передачи
    int repeat_count = 500; 
    int len_arr = base_len * repeat_count;
    int *bit_arr = (int*)malloc(len_arr * sizeof(int));
    
    for(int rep = 0; rep < repeat_count; rep++) {
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

    // Создаем буфер с преамбулой и данными
    int preamble_samples = 2000; // Длина преамбулы в сэмплах
    
    packet_cfg.data_len = conv_length;
    packet_cfg.total_len = preamble_samples + conv_length;
    
    total_samples = packet_cfg.total_len * 10; // 10 пакетов для непрерывной передачи
    tx_samples = (int16_t *)malloc(total_samples * 2 * sizeof(int16_t));
    int scale_factor = 3000;
    
    int offset = 0;
    
    // Генерируем несколько пакетов с преамбулой
    for(int packet = 0; packet < 10; packet++) {
        // Добавляем преамбулу
        static float phase = 0.0f;
        for(int i = 0; i < preamble_samples; i++) {
            float freq_offset = 0.0001;
            float cos_p = cosf(phase);
            float sin_p = sinf(phase);
            
            // Преамбула - тональный сигнал
            float s = cosf(2.0f * M_PI * i / 100);
            
            float i_val = s * cos_p;
            float q_val = s * sin_p;
            
            tx_samples[offset++] = (int16_t)(i_val * scale_factor);
            tx_samples[offset++] = (int16_t)(q_val * scale_factor);
            
            phase += freq_offset;
            if (phase > 2 * M_PI) phase -= 2 * M_PI;
        }
        
        // Добавляем данные
        for(int i = 0; i < conv_length; i++) {
            float freq_offset = 0.0001;
            float s = (float)conv_result[i];
            
            float cos_p = cosf(phase);
            float sin_p = sinf(phase);
            
            float i_val = s * cos_p;
            float q_val = s * sin_p;
            
            tx_samples[offset++] = (int16_t)(i_val * scale_factor);
            tx_samples[offset++] = (int16_t)(q_val * scale_factor);
            
            phase += freq_offset;
            if (phase > 2 * M_PI) phase -= 2 * M_PI;
        }
    }

    printf("TX: Generated %d total samples\n", total_samples);
    printf("TX: Packet structure: Preamble=%d, Data=%d samples\n", 
           preamble_samples, conv_length);
    printf("TX: Signal length: %.2f seconds\n", (float)total_samples / sample_rate);

    free(bit_arr);
    free(bpsk_arr);
    free(bpsk_after_arr);
    free(conv_result);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <sdr_uri>\n", argv[0]);
        printf("Example: %s usb:1.3.5\n", argv[0]);
        printf("Example: %s ip:192.168.2.1\n", argv[0]);
        return EXIT_FAILURE;
    }

    signal(SIGINT, sigint_handler);

    char* sdr_uri = argv[1];
    
    printf("BPSK Transmitter with Preamble\n");
    printf("SDR URI: %s\n", sdr_uri);
    printf("Sample rate: %d Hz\n", sample_rate);

    // Генерация тестового сигнала
    generate_test_signal();

    // Инициализация SDR
    SoapySDRKwargs args = {};
    SoapySDRKwargs_set(&args, "driver", "plutosdr");
    SoapySDRKwargs_set(&args, "uri", sdr_uri);
    
    printf("TX: Opening SDR...\n");
    SoapySDRDevice *sdr = SoapySDRDevice_make(&args);
    SoapySDRKwargs_clear(&args);

    if (!sdr) {
        printf("TX: ERROR - Failed to open SDR at %s\n", sdr_uri);
        free(tx_samples);
        return EXIT_FAILURE;
    }

    printf("TX: SDR opened successfully\n");

    // Настройка параметров
    double tx_freq = 800e6; // 800 MHz

    SoapySDRDevice_setSampleRate(sdr, SOAPY_SDR_TX, 0, sample_rate);
    SoapySDRDevice_setFrequency(sdr, SOAPY_SDR_TX, 0, tx_freq, NULL);
    SoapySDRDevice_setGain(sdr, SOAPY_SDR_TX, 0, 70.0); // Уменьшил усиление

    printf("TX: Frequency: %.3f MHz\n", tx_freq / 1e6);

    size_t channels[] = {0};
    const size_t channel_count = 1;
    
    printf("TX: Setting up stream...\n");
    SoapySDRStream *txStream = SoapySDRDevice_setupStream(sdr, SOAPY_SDR_TX, SOAPY_SDR_CS16, channels, channel_count, NULL);
    if (!txStream) {
        printf("TX: Failed to setup stream\n");
        SoapySDRDevice_unmake(sdr);
        free(tx_samples);
        return EXIT_FAILURE;
    }

    SoapySDRDevice_activateStream(sdr, txStream, 0, 0, 0);

    size_t tx_mtu = SoapySDRDevice_getStreamMTU(sdr, txStream);
    int16_t *tx_buff = (int16_t*)malloc(2 * tx_mtu * sizeof(int16_t));

    const long timeoutUs = 400000;
    long long tx_time = 0;
    int samples_sent = 0;

    printf("TX: Starting continuous transmission. Press Ctrl+C to stop.\n");
    printf("TX: MTU = %zu samples\n", tx_mtu);

    while (running) {
        int flags = SOAPY_SDR_HAS_TIME;
        
        // Определяем сколько сэмплов отправить
        int samples_to_send;
        if (total_samples - samples_sent < (int)tx_mtu) {
            samples_to_send = total_samples - samples_sent;
        } else {
            samples_to_send = (int)tx_mtu;
        }

        // Если дошли до конца буфера, начинаем сначала (loop)
        if (samples_to_send <= 0) {
            samples_sent = 0;
            samples_to_send = (int)tx_mtu;
            printf("TX: Looping signal...\n");
        }

        // Заполняем буфер для отправки
        for (int i = 0; i < samples_to_send * 2; i++) {
            tx_buff[i] = tx_samples[samples_sent * 2 + i];
        }

        // Если отправили меньше чем MTU, дополняем нулями
        for (int i = samples_to_send * 2; i < (int)tx_mtu * 2; i++) {
            tx_buff[i] = 0;
        }

        void *tx_buffs[] = {tx_buff};
        int ret = SoapySDRDevice_writeStream(sdr, txStream, (const void * const*)tx_buffs, tx_mtu, &flags, tx_time, timeoutUs);
        
        if (ret < 0) {
            printf("TX: Write stream error: %d\n", ret);
        } else {
            samples_sent += samples_to_send;
            tx_time += (samples_to_send * 1000000000LL) / sample_rate;
            
            // Печатаем прогресс каждые 50000 сэмплов
            if (samples_sent % 50000 == 0) {
                printf("TX: Sent %d samples (%.1f seconds)\n", 
                       samples_sent, (float)samples_sent / sample_rate);
            }
        }
    }

    printf("TX: Cleaning up...\n");
    free(tx_buff);
    free(tx_samples);

    SoapySDRDevice_deactivateStream(sdr, txStream, 0, 0);
    SoapySDRDevice_closeStream(sdr, txStream);
    SoapySDRDevice_unmake(sdr);
    
    printf("TX: Shutdown complete\n");
    return EXIT_SUCCESS;
}