/*
 * Minimal public C ABI declarations used by Omawake's whisper.cpp adapter.
 *
 * Derived from whisper.cpp v1.9.3 public headers at commit
 * 371b5a7561823ab2bb32142d2751e35e7534727b. This contains declarations and
 * data layouts only; it contains no whisper.cpp implementation code.
 * See LICENSE in this directory.
 */
#ifndef OMAWAKE_WHISPER_ABI_1_9_3_H
#define OMAWAKE_WHISPER_ABI_1_9_3_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define OMAWAKE_WHISPER_VERSION "1.9.3"

struct whisper_context;
struct whisper_state;
struct whisper_vad_context;

typedef int32_t whisper_token;

enum ggml_log_level {
    GGML_LOG_LEVEL_NONE = 0,
    GGML_LOG_LEVEL_DEBUG = 1,
    GGML_LOG_LEVEL_INFO = 2,
    GGML_LOG_LEVEL_WARN = 3,
    GGML_LOG_LEVEL_ERROR = 4,
    GGML_LOG_LEVEL_CONT = 5,
};

typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
typedef bool (*ggml_abort_callback)(void * data);

enum whisper_alignment_heads_preset {
    WHISPER_AHEADS_NONE,
    WHISPER_AHEADS_N_TOP_MOST,
    WHISPER_AHEADS_CUSTOM,
    WHISPER_AHEADS_TINY_EN,
    WHISPER_AHEADS_TINY,
    WHISPER_AHEADS_BASE_EN,
    WHISPER_AHEADS_BASE,
    WHISPER_AHEADS_SMALL_EN,
    WHISPER_AHEADS_SMALL,
    WHISPER_AHEADS_MEDIUM_EN,
    WHISPER_AHEADS_MEDIUM,
    WHISPER_AHEADS_LARGE_V1,
    WHISPER_AHEADS_LARGE_V2,
    WHISPER_AHEADS_LARGE_V3,
    WHISPER_AHEADS_LARGE_V3_TURBO,
};

typedef struct whisper_ahead {
    int n_text_layer;
    int n_head;
} whisper_ahead;

typedef struct whisper_aheads {
    size_t n_heads;
    const whisper_ahead * heads;
} whisper_aheads;

struct whisper_context_params {
    bool use_gpu;
    bool flash_attn;
    int gpu_device;
    bool dtw_token_timestamps;
    enum whisper_alignment_heads_preset dtw_aheads_preset;
    int dtw_n_top;
    struct whisper_aheads dtw_aheads;
    size_t dtw_mem_size;
};

typedef struct whisper_token_data whisper_token_data;

enum whisper_gretype {
    WHISPER_GRETYPE_END = 0,
    WHISPER_GRETYPE_ALT = 1,
    WHISPER_GRETYPE_RULE_REF = 2,
    WHISPER_GRETYPE_CHAR = 3,
    WHISPER_GRETYPE_CHAR_NOT = 4,
    WHISPER_GRETYPE_CHAR_RNG_UPPER = 5,
    WHISPER_GRETYPE_CHAR_ALT = 6,
};

typedef struct whisper_grammar_element {
    enum whisper_gretype type;
    uint32_t value;
} whisper_grammar_element;

typedef struct whisper_vad_params {
    float threshold;
    int min_speech_duration_ms;
    int min_silence_duration_ms;
    float max_speech_duration_s;
    int speech_pad_ms;
    float samples_overlap;
} whisper_vad_params;

enum whisper_sampling_strategy {
    WHISPER_SAMPLING_GREEDY,
    WHISPER_SAMPLING_BEAM_SEARCH,
};

typedef void (*whisper_new_segment_callback)(struct whisper_context *, struct whisper_state *, int, void *);
typedef void (*whisper_progress_callback)(struct whisper_context *, struct whisper_state *, int, void *);
typedef bool (*whisper_encoder_begin_callback)(struct whisper_context *, struct whisper_state *, void *);
typedef void (*whisper_logits_filter_callback)(
        struct whisper_context *,
        struct whisper_state *,
        const whisper_token_data *,
        int,
        float *,
        void *);

struct whisper_full_params {
    enum whisper_sampling_strategy strategy;
    int n_threads;
    int n_max_text_ctx;
    int offset_ms;
    int duration_ms;
    bool translate;
    bool no_context;
    bool no_timestamps;
    bool single_segment;
    bool print_special;
    bool print_progress;
    bool print_realtime;
    bool print_timestamps;
    bool token_timestamps;
    float thold_pt;
    float thold_ptsum;
    int max_len;
    bool split_on_word;
    int max_tokens;
    bool debug_mode;
    int audio_ctx;
    bool tdrz_enable;
    const char * suppress_regex;
    const char * initial_prompt;
    bool carry_initial_prompt;
    const whisper_token * prompt_tokens;
    int prompt_n_tokens;
    const char * language;
    bool detect_language;
    bool suppress_blank;
    bool suppress_nst;
    float temperature;
    float max_initial_ts;
    float length_penalty;
    float temperature_inc;
    float entropy_thold;
    float logprob_thold;
    float no_speech_thold;
    struct {
        int best_of;
    } greedy;
    struct {
        int beam_size;
        float patience;
    } beam_search;
    whisper_new_segment_callback new_segment_callback;
    void * new_segment_callback_user_data;
    whisper_progress_callback progress_callback;
    void * progress_callback_user_data;
    whisper_encoder_begin_callback encoder_begin_callback;
    void * encoder_begin_callback_user_data;
    ggml_abort_callback abort_callback;
    void * abort_callback_user_data;
    whisper_logits_filter_callback logits_filter_callback;
    void * logits_filter_callback_user_data;
    const whisper_grammar_element ** grammar_rules;
    size_t n_grammar_rules;
    size_t i_start_rule;
    float grammar_penalty;
    bool vad;
    const char * vad_model_path;
    whisper_vad_params vad_params;
};

struct whisper_vad_context_params {
    int n_threads;
    bool use_gpu;
    int gpu_device;
};

/* Guard the LP64 layouts used by the Linux x86_64 and aarch64 packages. */
#if defined(__LP64__)
_Static_assert(sizeof(struct whisper_context_params) == 48, "unexpected whisper_context_params layout");
_Static_assert(sizeof(struct whisper_full_params) == 304, "unexpected whisper_full_params layout");
_Static_assert(sizeof(struct whisper_vad_context_params) == 12, "unexpected whisper_vad_context_params layout");
#endif

#endif
