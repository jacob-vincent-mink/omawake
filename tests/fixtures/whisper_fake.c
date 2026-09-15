/* Safe test double for the public whisper.cpp symbols Omawake uses. */
#include "whisper_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float vad_probability = 0.9f;

#ifndef OMAWAKE_FAKE_VERSION
#define OMAWAKE_FAKE_VERSION OMAWAKE_WHISPER_VERSION
#endif

static void record(const char * event) {
    const char * path = getenv("OMAWAKE_FAKE_WHISPER_LOG");
    if (path == NULL) {
        return;
    }
    FILE * output = fopen(path, "a");
    if (output != NULL) {
        fprintf(output, "%s\n", event);
        fclose(output);
    }
}

static void exit_worker_once(void) {
    const char * marker = getenv("OMAWAKE_FAKE_WHISPER_EXIT_ONCE");
    if (marker == NULL) {
        return;
    }
    FILE * existing = fopen(marker, "r");
    if (existing != NULL) {
        fclose(existing);
        return;
    }
    FILE * created = fopen(marker, "w");
    if (created != NULL) {
        fclose(created);
        _Exit(0);
    }
}

const char * whisper_version(void) {
    return OMAWAKE_FAKE_VERSION;
}

void whisper_log_set(ggml_log_callback callback, void * user_data) {
    (void) callback;
    (void) user_data;
}

struct whisper_context_params whisper_context_default_params(void) {
    struct whisper_context_params params = {0};
    return params;
}

struct whisper_context * whisper_init_from_file_with_params(
        const char * path,
        struct whisper_context_params params) {
    (void) path;
    (void) params;
    record("verifier-load");
    return (struct whisper_context *) calloc(1, 1);
}

struct whisper_full_params whisper_full_default_params(enum whisper_sampling_strategy strategy) {
    struct whisper_full_params params = {0};
    params.strategy = strategy;
    return params;
}

int whisper_full(
        struct whisper_context * context,
        struct whisper_full_params params,
        const float * samples,
        int sample_count) {
    (void) context;
    (void) params;
    (void) samples;
    (void) sample_count;
    record("transcribe");
    return 0;
}

int whisper_full_n_segments(struct whisper_context * context) {
    (void) context;
    return 1;
}

const char * whisper_full_get_segment_text(struct whisper_context * context, int segment) {
    (void) context;
    (void) segment;
    return "computer";
}

void whisper_free(struct whisper_context * context) {
    free(context);
}

struct whisper_vad_context_params whisper_vad_default_context_params(void) {
    struct whisper_vad_context_params params = {0};
    return params;
}

struct whisper_vad_context * whisper_vad_init_from_file_with_params(
        const char * path,
        struct whisper_vad_context_params params) {
    (void) path;
    (void) params;
    record("vad-load");
    return (struct whisper_vad_context *) calloc(1, 1);
}

bool whisper_vad_detect_speech_no_reset(
        struct whisper_vad_context * context,
        const float * samples,
        int sample_count) {
    (void) context;
    (void) samples;
    return sample_count == 512;
}

void whisper_vad_reset_state(struct whisper_vad_context * context) {
    (void) context;
    exit_worker_once();
}

int whisper_vad_n_probs(struct whisper_vad_context * context) {
    (void) context;
    return 1;
}

float * whisper_vad_probs(struct whisper_vad_context * context) {
    (void) context;
    return &vad_probability;
}

void whisper_vad_free(struct whisper_vad_context * context) {
    free(context);
}
