#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "whisper_abi.h"

#ifndef OMA_WHISPER_ABI_VERSION
#error "OMA_WHISPER_ABI_VERSION must match the whisper.h used to build this adapter"
#endif

struct oma_whisper_api {
    const char * (*version)(void);
    void (*log_set)(ggml_log_callback, void *);
    struct whisper_context_params (*context_default_params)(void);
    struct whisper_context * (*init_from_file_with_params)(const char *, struct whisper_context_params);
    struct whisper_full_params (*full_default_params)(enum whisper_sampling_strategy);
    int (*full)(struct whisper_context *, struct whisper_full_params, const float *, int);
    int (*full_n_segments)(struct whisper_context *);
    const char * (*full_get_segment_text)(struct whisper_context *, int);
    void (*free_context)(struct whisper_context *);
    struct whisper_vad_context_params (*vad_default_context_params)(void);
    struct whisper_vad_context * (*vad_init_from_file_with_params)(const char *, struct whisper_vad_context_params);
    bool (*vad_detect_speech_no_reset)(struct whisper_vad_context *, const float *, int);
    void (*vad_reset_state)(struct whisper_vad_context *);
    int (*vad_n_probs)(struct whisper_vad_context *);
    float * (*vad_probs)(struct whisper_vad_context *);
    void (*vad_free)(struct whisper_vad_context *);
};

struct oma_whisper {
    void * library;
    struct whisper_context * verifier;
    struct whisper_vad_context * vad;
    struct oma_whisper_api api;
    int n_threads;
    char version[64];
};

static void silent_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static void set_error(char * error, size_t capacity, const char * message) {
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message == NULL ? "unknown error" : message);
    }
}

static int resolve_symbol(
        void * library,
        const char * name,
        void ** output,
        char * error,
        size_t error_capacity) {
    dlerror();
    *output = dlsym(library, name);
    const char * reason = dlerror();
    if (reason == NULL && *output != NULL) {
        return 0;
    }
    char message[512];
    snprintf(message, sizeof(message), "libwhisper is missing required symbol %s: %s", name,
             reason == NULL ? "symbol resolved to null" : reason);
    set_error(error, error_capacity, message);
    return -1;
}

#define RESOLVE(api, library, member, symbol, error, capacity) \
    do { \
        if (resolve_symbol((library), (symbol), (void **) &(api)->member, (error), (capacity)) != 0) { \
            return -1; \
        } \
    } while (0)

static int resolve_api(struct oma_whisper_api * api, void * library, char * error, size_t capacity) {
    memset(api, 0, sizeof(*api));
    RESOLVE(api, library, version, "whisper_version", error, capacity);
    RESOLVE(api, library, log_set, "whisper_log_set", error, capacity);
    RESOLVE(api, library, context_default_params, "whisper_context_default_params", error, capacity);
    RESOLVE(api, library, init_from_file_with_params, "whisper_init_from_file_with_params", error, capacity);
    RESOLVE(api, library, full_default_params, "whisper_full_default_params", error, capacity);
    RESOLVE(api, library, full, "whisper_full", error, capacity);
    RESOLVE(api, library, full_n_segments, "whisper_full_n_segments", error, capacity);
    RESOLVE(api, library, full_get_segment_text, "whisper_full_get_segment_text", error, capacity);
    RESOLVE(api, library, free_context, "whisper_free", error, capacity);
    RESOLVE(api, library, vad_default_context_params, "whisper_vad_default_context_params", error, capacity);
    RESOLVE(api, library, vad_init_from_file_with_params, "whisper_vad_init_from_file_with_params", error, capacity);
    RESOLVE(api, library, vad_detect_speech_no_reset, "whisper_vad_detect_speech_no_reset", error, capacity);
    RESOLVE(api, library, vad_reset_state, "whisper_vad_reset_state", error, capacity);
    RESOLVE(api, library, vad_n_probs, "whisper_vad_n_probs", error, capacity);
    RESOLVE(api, library, vad_probs, "whisper_vad_probs", error, capacity);
    RESOLVE(api, library, vad_free, "whisper_vad_free", error, capacity);
    return 0;
}

static void close_provider(struct oma_whisper * provider) {
    if (provider == NULL) {
        return;
    }
    if (provider->vad != NULL) {
        provider->api.vad_free(provider->vad);
    }
    if (provider->verifier != NULL) {
        provider->api.free_context(provider->verifier);
    }
    if (provider->library != NULL) {
        dlclose(provider->library);
    }
    free(provider);
}

int oma_whisper_open(
        const char * library_path,
        const char * verifier_path,
        const char * vad_path,
        int n_threads,
        struct oma_whisper ** output,
        char * error,
        size_t error_capacity) {
    if (library_path == NULL || verifier_path == NULL || vad_path == NULL || output == NULL) {
        set_error(error, error_capacity, "library, verifier, VAD, and output are required");
        return -1;
    }
    *output = NULL;
    void * library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        char message[1024];
        snprintf(message, sizeof(message), "load libwhisper %s: %s", library_path, dlerror());
        set_error(error, error_capacity, message);
        return -1;
    }
    struct oma_whisper * provider = calloc(1, sizeof(*provider));
    if (provider == NULL) {
        set_error(error, error_capacity, strerror(errno));
        dlclose(library);
        return -1;
    }
    provider->library = library;
    provider->n_threads = n_threads > 0 ? n_threads : 1;
    if (resolve_api(&provider->api, library, error, error_capacity) != 0) {
        close_provider(provider);
        return -1;
    }
    const char * version = provider->api.version();
    if (version == NULL || strcmp(version, OMA_WHISPER_ABI_VERSION) != 0) {
        char message[512];
        snprintf(message, sizeof(message), "unsupported libwhisper ABI version %s; adapter was built for %s",
                 version == NULL ? "<null>" : version, OMA_WHISPER_ABI_VERSION);
        set_error(error, error_capacity, message);
        close_provider(provider);
        return -1;
    }
    snprintf(provider->version, sizeof(provider->version), "%s", version);
    provider->api.log_set(silent_log, NULL);

    struct whisper_context_params verifier_params = provider->api.context_default_params();
    verifier_params.use_gpu = false;
    provider->verifier = provider->api.init_from_file_with_params(verifier_path, verifier_params);
    if (provider->verifier == NULL) {
        char message[1024];
        snprintf(message, sizeof(message), "libwhisper %s failed to load verifier %s", provider->version, verifier_path);
        set_error(error, error_capacity, message);
        close_provider(provider);
        return -1;
    }

    struct whisper_vad_context_params vad_params = provider->api.vad_default_context_params();
    vad_params.n_threads = provider->n_threads;
    vad_params.use_gpu = false;
    provider->vad = provider->api.vad_init_from_file_with_params(vad_path, vad_params);
    if (provider->vad == NULL) {
        char message[1024];
        snprintf(message, sizeof(message), "libwhisper %s failed to load VAD %s", provider->version, vad_path);
        set_error(error, error_capacity, message);
        close_provider(provider);
        return -1;
    }
    *output = provider;
    return 0;
}

const char * oma_whisper_version(const struct oma_whisper * provider) {
    return provider == NULL ? NULL : provider->version;
}

int oma_whisper_vad_probability(
        struct oma_whisper * provider,
        const float * samples,
        int sample_count,
        float * probability,
        char * error,
        size_t error_capacity) {
    if (provider == NULL || samples == NULL || sample_count != 512 || probability == NULL) {
        set_error(error, error_capacity, "VAD requires one 512-sample frame and an output");
        return -1;
    }
    if (!provider->api.vad_detect_speech_no_reset(provider->vad, samples, sample_count)) {
        set_error(error, error_capacity, "whisper VAD inference failed");
        return -1;
    }
    const int probabilities = provider->api.vad_n_probs(provider->vad);
    float * values = provider->api.vad_probs(provider->vad);
    if (probabilities != 1 || values == NULL) {
        set_error(error, error_capacity, "whisper VAD returned an unexpected probability shape");
        return -1;
    }
    *probability = values[0];
    return 0;
}

void oma_whisper_vad_reset(struct oma_whisper * provider) {
    if (provider != NULL && provider->vad != NULL) {
        provider->api.vad_reset_state(provider->vad);
    }
}

int oma_whisper_transcribe(
        struct oma_whisper * provider,
        const float * samples,
        int sample_count,
        const char * prompt,
        char * text,
        size_t text_capacity,
        char * error,
        size_t error_capacity) {
    if (provider == NULL || samples == NULL || sample_count <= 0 || text == NULL || text_capacity == 0) {
        set_error(error, error_capacity, "provider, non-empty PCM, and output buffer are required");
        return -1;
    }
    struct whisper_full_params params = provider->api.full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = provider->n_threads;
    params.language = "en";
    params.no_context = true;
    params.single_segment = true;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.initial_prompt = prompt;
    if (provider->api.full(provider->verifier, params, samples, sample_count) != 0) {
        set_error(error, error_capacity, "whisper_full failed");
        return -1;
    }
    text[0] = '\0';
    size_t used = 0;
    const int segments = provider->api.full_n_segments(provider->verifier);
    for (int index = 0; index < segments; ++index) {
        const char * segment = provider->api.full_get_segment_text(provider->verifier, index);
        if (segment == NULL) {
            continue;
        }
        const size_t length = strlen(segment);
        if (length >= text_capacity - used) {
            set_error(error, error_capacity, "transcript exceeds caller output buffer");
            return -1;
        }
        memcpy(text + used, segment, length);
        used += length;
        text[used] = '\0';
    }
    return segments;
}

void oma_whisper_close(struct oma_whisper * provider) {
    close_provider(provider);
}
