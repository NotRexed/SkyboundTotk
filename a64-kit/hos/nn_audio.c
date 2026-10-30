/* nn::audio: an HLE of the audio renderer at the API level, and its output.

   The game builds its sound graph through nn::audio calls on objects it owns (VoiceType, SubMixType, FinalMixType,
   SplitterType, DeviceSinkType, MemoryPoolType, effect types): each gets our handle in its first 8 bytes (AU_MAGIC,
   kind, index). A mixer thread renders one audio frame (240 samples at 48 kHz) every 5 ms:
     voices (PCM16, PCM float, DSP-ADPCM; resampled by sample rate x pitch, voice volume)
       -> their destination mix or splitter (per channel pair volumes)
       -> sub mixes in dependency order (sub mix volume, then to their destination)
       -> the final mix (volume) -> the device sink (its channel map, down to stereo) -> waveOut
   and signals the renderer event the game's audio thread waits on. Effects are accepted and left out (dry signal);
   aux effects pass through. Played sample counts and released wave buffers go back to the game as on hardware.
   The game's parameter changes take effect together at RequestUpdateAudioRenderer, and volume changes ramp over a
   frame, as the console's renderer does them (see "parameter updates").
   HOS_AUDIO_DUMP=<file.wav>: also write what is played (checking without speakers); HOS_AUDIO=0: silence. */
#include "hos.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>

void hos_system_event_init(uint64_t key, int autoclear);
void hos_system_event_signal(uint64_t key);

#define FRAME 240            /* samples per audio frame (5 ms at 48 kHz) */
#define RATE 48000
#define AU_MAGIC 0x41554448u /* the high word of our handles in the game's objects */
#define MAX_VOICES 512
#define MAX_MIXES 96
#define MAX_SPLITTERS 128
#define MAX_SPLIT_DESTS 32
#define MAX_CH 8             /* channels of a voice */
#define MAX_BUF 24           /* buffers of a mix */
#define MAX_WB 32            /* queued wave buffers per voice */
#define MAX_REL 128

enum { K_VOICE = 1, K_MIX, K_SPLITTER, K_SINK, K_POOL, K_EFFECT, K_CONFIG };
enum { FMT_INVALID, FMT_PCM8, FMT_PCM16, FMT_PCM24, FMT_PCM32, FMT_FLOAT, FMT_ADPCM };
enum { PLAY_STARTED, PLAY_STOPPED, PLAY_PAUSED };
enum { DEST_NONE, DEST_MIX, DEST_SPLITTER };
typedef struct { uint64_t buffer, size; int32_t start, end; int loop; uint64_t context; } WB;

typedef struct {
    int used, released, final_mix, buffers;    /* (released: by the game, until the next update applies it) */
    float volume;
    int dest_kind, dest;                       /* (sub mixes) */
    float to[MAX_BUF][MAX_BUF];                /* [src buffer][dst buffer] volume to the destination mix */
    float buf[MAX_BUF][FRAME];
    int depth;                                 /* (processing order: deeper first) */
} Mix;

typedef struct {
    int used, released, channels, dests;
    int dest_mix[MAX_SPLIT_DESTS];
    float vol[MAX_SPLIT_DESTS][MAX_CH][MAX_BUF];
    float ramp[MAX_SPLIT_DESTS][MAX_CH][MAX_BUF];   /* the volumes the last frame ended with */
} Splitter;

typedef struct {
    int used, released, rate, channels, format, state;
    uint64_t guest, param;                     /* the VoiceType; AdpcmParameter */
    int16_t coefs[16];                         /* ADPCM coefficients, copied when acquired (the game reuses the
                                                  parameter memory) */
    float volume, pitch;
    int dest_kind, dest;
    float vol[MAX_CH][MAX_BUF];                /* to the destination mix */
    float ramp_volume, ramp_vol[MAX_CH][MAX_BUF];   /* the volumes the last frame ended with (ramped to the above) */
    int snap;                                  /* started or a new destination: no ramp from the old volumes */
    uint64_t wb[MAX_WB];                       /* queued WaveBuffer*, oldest first (their identity) */
    WB wbc[MAX_WB];                            /* and their contents when appended */
    int nwb, pending_wb;                       /* (pending: appended, not yet applied by an update) */
    uint64_t rel[MAX_REL];                     /* released, for GetReleasedWaveBuffer */
    int nrel;
    int64_t played;
    int64_t pos;                               /* sample frame within the current wave buffer */
    int started_wb;                            /* the current one's start was set up */
    double frac;
    float prev[MAX_CH], cur[MAX_CH];           /* resampler: the last two source frames */
    int16_t yn1, yn2;                          /* ADPCM history (mono) */
    int bq_on[2];                              /* the voice's two biquad filters (BiquadFilterParameter, Q14) */
    float bq_b[2][3], bq_a[2][2], bq_z[2][MAX_CH][2];
    int dropped;
} Voice;

static CRITICAL_SECTION lock;
static int lock_ready;
static Voice voices[MAX_VOICES];
static Mix mixes[MAX_MIXES];
static Splitter splitters[MAX_SPLITTERS];
static int sink_mix = -1, sink_n;
static float lim_in = 1.0f, lim_out = 1.0f, lim_threshold = 1.0f, lim_env = 1.0f;   /* the light limiter */
static int lim_enabled;
static int8_t sink_in[8];

static void au_lock(void) {
    if (!lock_ready) { InitializeCriticalSection(&lock); lock_ready = 1; }
    EnterCriticalSection(&lock);
}
static void au_unlock(void) { LeaveCriticalSection(&lock); }

static void put_handle(uint64_t obj, int kind, int index) {
    if (obj) G64(obj) = (uint64_t)AU_MAGIC << 32 | (uint64_t)kind << 24 | (uint32_t)index;
}
static int au_trace = -1;   /* HOS_AUDIO_TRACE=1: the graph setup calls */
#define AUT(...) do { if (au_trace < 0) au_trace = getenv("HOS_AUDIO_TRACE") != NULL; if (au_trace) hos_log(__VA_ARGS__); } while (0)
static int get_handle(uint64_t obj, int kind) {
    if (!obj) return -1;
    const uint64_t h = G64(obj);
    if ((uint32_t)(h >> 32) != AU_MAGIC || (int)((h >> 24) & 0xFF) != kind) return -1;
    return (int)(h & 0xFFFFFF);
}
static float farg(Ctx *c, int n) {
    float f;
    memcpy(&f, &c->vreg[n][0], 4);
    return f;
}
/* The game's objects, by the handle in their first word; what the game released (until an update applies it) no
   longer counts. */
static int voice_index(uint64_t obj) {
    const int i = get_handle(obj, K_VOICE);
    return i >= 0 && i < MAX_VOICES && voices[i].used && !voices[i].released ? i : -1;
}
static Voice *voice_of(uint64_t obj) {
    const int i = voice_index(obj);
    return i >= 0 ? &voices[i] : NULL;
}
static int mix_index(uint64_t obj) {
    const int i = get_handle(obj, K_MIX);
    return i >= 0 && i < MAX_MIXES && mixes[i].used && !mixes[i].released ? i : -1;
}
static Mix *mix_of(uint64_t obj) {
    const int i = mix_index(obj);
    return i >= 0 ? &mixes[i] : NULL;
}
static int splitter_index(uint64_t obj) {
    const int i = get_handle(obj, K_SPLITTER);
    return i >= 0 && i < MAX_SPLITTERS && splitters[i].used && !splitters[i].released ? i : -1;
}
static Splitter *splitter_of(uint64_t obj) {
    const int i = splitter_index(obj);
    return i >= 0 ? &splitters[i] : NULL;
}

/* ---- wave buffers ---------------------------------------------------------------------------------------------
   nn::audio::WaveBuffer: buffer @0, size @8, startSampleOffset @0x10, endSampleOffset @0x14, loop @0x18,
   isEndOfStream @0x19, pContext @0x20 (AdpcmContext: predScale, history[2]), contextSize @0x28. Copied when
   appended (TOTK appends every buffer through one struct it refills); the pointer is only its identity for
   GetReleasedWaveBuffer. */
static WB read_wb(uint64_t p) {
    WB w;
    w.buffer = G64(p);
    w.size = G64(p + 8);
    w.start = (int32_t)G32(p + 0x10);
    w.end = (int32_t)G32(p + 0x14);
    w.loop = *(uint8_t *)GP(p + 0x18);
    w.context = G64(p + 0x20);
    return w;
}

/* one source frame (all channels) of the voice's current wave buffer at sample frame s; 0 when out of range */
static void adpcm_seek(Voice *v, const WB *w);
static int16_t adpcm_sample(Voice *v, const WB *w, int64_t s) {
    /* 8-byte frames of 14 samples: a header byte (coefficient pair index in 4-6, scale exponent in 0-3), then
       nibbles high first. Decoded in order: s must follow the previous call (adpcm_seek resets it). */
    const uint8_t *data = (const uint8_t *)GP(w->buffer);
    const int64_t frame = s / 14, in = s % 14;
    if ((uint64_t)(frame * 8 + 8) > w->size) return 0;
    const uint8_t header = data[frame * 8];
    const uint8_t byte = data[frame * 8 + 1 + in / 2];
    int nibble = in & 1 ? byte & 0xF : byte >> 4;
    if (nibble >= 8) nibble -= 16;
    const int idx = (header >> 4) & 7, scale = 1 << (header & 0xF);
    const int c1 = v->coefs[idx * 2], c2 = v->coefs[idx * 2 + 1];
    int32_t val = ((nibble * scale) << 11) + c1 * v->yn1 + c2 * v->yn2;
    val = (val + 1024) >> 11;
    if (val > 32767) val = 32767;
    if (val < -32768) val = -32768;
    v->yn2 = v->yn1;
    v->yn1 = (int16_t)val;
    return (int16_t)val;
}
static void adpcm_seek(Voice *v, const WB *w) {
    if (w->context) {   /* AdpcmContext { u16 predScale; s16 history[2]; } */
        const int16_t *ctx = (const int16_t *)GP(w->context);
        v->yn1 = ctx[1];
        v->yn2 = ctx[2];
    }
}
static void source_frame(Voice *v, const WB *w, int64_t s, float *out) {
    for (int ch = 0; ch < v->channels; ch++) out[ch] = 0;
    switch (v->format) {
    case FMT_PCM16: {
        const uint64_t off = (uint64_t)s * v->channels * 2;
        if (off + (uint64_t)v->channels * 2 > w->size) return;
        const int16_t *p = (const int16_t *)GP(w->buffer + off);
        for (int ch = 0; ch < v->channels; ch++) out[ch] = p[ch] * (1.0f / 32768.0f);
        break;
    }
    case FMT_FLOAT: {
        const uint64_t off = (uint64_t)s * v->channels * 4;
        if (off + (uint64_t)v->channels * 4 > w->size) return;
        const float *p = (const float *)GP(w->buffer + off);
        for (int ch = 0; ch < v->channels; ch++) out[ch] = p[ch];
        break;
    }
    case FMT_PCM8: {
        const uint64_t off = (uint64_t)s * v->channels;
        if (off + (uint64_t)v->channels > w->size) return;
        const int8_t *p = (const int8_t *)GP(w->buffer + off);
        for (int ch = 0; ch < v->channels; ch++) out[ch] = p[ch] * (1.0f / 128.0f);
        break;
    }
    case FMT_ADPCM:
        out[0] = adpcm_sample(v, w, s) * (1.0f / 32768.0f);
        break;
    default:
        break;
    }
}

static void release_wb(Voice *v, uint64_t wb) {
    if (v->nrel == MAX_REL) {
        memmove(v->rel, v->rel + 1, (size_t)(MAX_REL - 1) * sizeof v->rel[0]);
        v->nrel--;
    }
    v->rel[v->nrel++] = wb;
}

/* the next source frame of the voice into v->cur; 0 when it ran out of wave buffers */
static int next_source_frame(Voice *v) {
    for (int guard = 0; guard < MAX_WB + 1; guard++) {
        if (!v->nwb) return 0;
        const WB w = v->wbc[0];
        if (!v->started_wb) {
            v->pos = w.start;
            v->started_wb = 1;
            if (v->format == FMT_ADPCM) {
                adpcm_seek(v, &w);
                for (int64_t s = (w.start / 14) * 14; s < w.start; s++) adpcm_sample(v, &w, s);   /* (mid-frame start) */
            }
        }
        if (v->pos < w.end) {
            float f[MAX_CH];
            source_frame(v, &w, v->pos, f);
            memcpy(v->cur, f, sizeof f);
            v->pos++;
            v->played++;
            return 1;
        }
        if (w.loop) {   /* (looping: back to the start, the same buffer stays) */
            v->started_wb = 0;
            if (w.end <= w.start) return 0;
            continue;
        }
        /* this buffer is done: released for the game */
        release_wb(v, v->wb[0]);
        memmove(v->wb, v->wb + 1, (size_t)(v->nwb - 1) * sizeof v->wb[0]);
        memmove(v->wbc, v->wbc + 1, (size_t)(v->nwb - 1) * sizeof v->wbc[0]);
        v->nwb--;
        v->started_wb = 0;
    }
    return 0;
}

/* ---- I3DL2 reverb (on a sub mix: its input buffers in place to its output buffers) ------------------------------
   Parameters as the game sets them (gains in millibels, times as nn::TimeSpan); the processing is ours, a small
   feedback delay network in the I3DL2 spirit: a pre-delay line feeding early reflection taps, then 4 delay lines
   (Hadamard feedback, high-frequency damping from the HF decay ratio), wet = room gain x (reflections + late),
   out = dry gain x in + wet. HOS_AUDIO_NO_REVERB=1: left out (dry). */
#define MAX_EFFECTS 16
#define REV_PRE (RATE / 2)          /* pre-delay line: 0.5 s */
#define REV_LINE 4096               /* each late line, at most ~85 ms */
typedef struct {
    int used, mix, enabled, n;
    int8_t in[MAX_CH], out[MAX_CH];
    float room, room_hf, refl, reverb, hf_ratio, diffusion, density, hf_ref, dry;   /* (mB for the gains) */
    int64_t decay_ns, refl_delay_ns, late_delay_ns;
    float pre[REV_PRE];
    int pre_pos;
    float line[4][REV_LINE];
    int pos[4], len[4];
    float lp[4];                    /* damping filter states */
} Reverb;
static Reverb reverbs[MAX_EFFECTS];
static float mb_gain(float mb) { return powf(10.0f, mb / 2000.0f); }
static void reverb_frame(Reverb *rv, Mix *m) {
    static int off = -1;
    if (off < 0) off = getenv("HOS_AUDIO_NO_REVERB") != NULL;
    if (off || !rv->enabled || rv->n <= 0) return;
    static const int base_len[4] = {1559, 1617, 1491, 1422};
    const float density = rv->density > 0 ? rv->density / 100.0f : 1.0f;
    for (int k = 0; k < 4; k++) {
        const int L = (int)(base_len[k] * (0.6f + 0.4f * density));
        rv->len[k] = L < 64 ? 64 : L >= REV_LINE ? REV_LINE - 1 : L;
        if (rv->pos[k] >= rv->len[k]) rv->pos[k] = 0;
    }
    const float decay = rv->decay_ns > 0 ? (float)rv->decay_ns / 1e9f : 1.5f;
    const float hf_ratio = rv->hf_ratio > 0.05f ? rv->hf_ratio : 0.5f;
    const float hf_ref = rv->hf_ref > 0 ? rv->hf_ref : 5000.0f;
    /* damping: a one-pole low pass whose cutoff falls with the HF decay ratio */
    const float cutoff = fminf(hf_ref * (0.5f + hf_ratio), 20000.0f);
    const float damp = expf(-2.0f * 3.14159265f * cutoff / RATE);
    const float room = rv->room > -10000 ? mb_gain(rv->room) : 0.0f;
    const float refl = rv->refl > -10000 ? mb_gain(rv->refl) : 0.0f;
    const float late = rv->reverb > -10000 ? mb_gain(rv->reverb) : 0.0f;
    const float dry = rv->dry;
    int pre_d = (int)((double)rv->refl_delay_ns * RATE / 1e9);
    int late_d = pre_d + (int)((double)rv->late_delay_ns * RATE / 1e9);
    if (pre_d < 0) pre_d = 0;
    if (pre_d >= REV_PRE / 2) pre_d = REV_PRE / 2 - 1;
    if (late_d >= REV_PRE) late_d = REV_PRE - 1;
    float g[4];
    for (int k = 0; k < 4; k++) g[k] = powf(10.0f, -3.0f * rv->len[k] / (RATE * decay));
    static const int er_tap[6] = {113, 397, 709, 1013, 1301, 1663};
    static const float er_gain[6] = {0.84f, -0.71f, 0.62f, -0.52f, 0.43f, -0.35f};
    const float diff = rv->diffusion > 0 ? fminf(rv->diffusion / 100.0f, 1.0f) : 1.0f;
    for (int i = 0; i < FRAME; i++) {
        float x = 0;
        for (int c = 0; c < rv->n; c++) {
            const int b = rv->in[c];
            if (b >= 0 && b < m->buffers) x += m->buf[b][i];
        }
        x /= (float)rv->n;
        if (!isfinite(x)) x = 0;   /* (a NaN would stay in the feedback lines for good) */
        rv->pre[rv->pre_pos] = x;
        float er = 0;
        for (int t = 0; t < 6; t++)
            er += er_gain[t] * rv->pre[(rv->pre_pos - pre_d - er_tap[t] + 2 * REV_PRE) % REV_PRE];
        er *= 0.4f;
        const float into_late = rv->pre[(rv->pre_pos - late_d + REV_PRE) % REV_PRE];
        rv->pre_pos = (rv->pre_pos + 1) % REV_PRE;
        float o[4];
        for (int k = 0; k < 4; k++) o[k] = rv->line[k][rv->pos[k]];
        /* Hadamard mix (orthonormal), damping, decay */
        const float hm[4] = {(o[0] + o[1] + o[2] + o[3]) * 0.5f, (o[0] - o[1] + o[2] - o[3]) * 0.5f,
                             (o[0] + o[1] - o[2] - o[3]) * 0.5f, (o[0] - o[1] - o[2] + o[3]) * 0.5f};
        for (int k = 0; k < 4; k++) {
            const float v = diff * hm[k] + (1.0f - diff) * o[k];
            rv->lp[k] = v * (1.0f - damp) + rv->lp[k] * damp;
            rv->line[k][rv->pos[k]] = into_late * 0.5f + rv->lp[k] * g[k];
            rv->pos[k] = (rv->pos[k] + 1) % rv->len[k];
        }
        for (int c = 0; c < rv->n; c++) {
            const int bi = rv->in[c], bo = rv->out[c];
            if (bo < 0 || bo >= m->buffers) continue;
            const float in_s = bi >= 0 && bi < m->buffers ? m->buf[bi][i] : 0.0f;
            const float sign = (c & 1) ? -1.0f : 1.0f;
            const float l = sign * o[c % 4] + o[(c + 1) % 4] * 0.5f;
            m->buf[bo][i] = dry * in_s + room * (refl * er * sign + late * l * 0.7f);
        }
    }
}
static void effects_frame(int mix) {
    for (int e = 0; e < MAX_EFFECTS; e++)
        if (reverbs[e].used && reverbs[e].mix == mix) reverb_frame(&reverbs[e], &mixes[mix]);
}

/* ---- the mixer ------------------------------------------------------------------------------------------------- */
static void render_voice(Voice *v) {
    /* (a started voice without a destination still plays, as on the DSP: its buffers run out and are released) */
    if (v->state != PLAY_STARTED) return;
    if (v->snap) {   /* (just started, or a new destination: its volumes at once) */
        v->snap = 0;
        v->ramp_volume = v->volume;
        memcpy(v->ramp_vol, v->vol, sizeof v->ramp_vol);
        if (v->dest_kind == DEST_SPLITTER && splitters[v->dest].used)
            memcpy(splitters[v->dest].ramp, splitters[v->dest].vol, sizeof splitters[v->dest].ramp);
    }
    const double step = (double)v->rate * (v->pitch > 0 ? v->pitch : 1.0) / RATE;
    float out[MAX_CH][FRAME];
    int any = 0;
    /* volumes ramp from the last frame's to the new ones over the frame, as the console's DSP does (a jump clicks) */
    const float vol0 = v->ramp_volume, dvol = (v->volume - v->ramp_volume) / FRAME;
    for (int i = 0; i < FRAME; i++) {
        v->frac += step;
        while (v->frac >= 1.0) {
            v->frac -= 1.0;
            memcpy(v->prev, v->cur, sizeof v->prev);
            if (!next_source_frame(v)) memset(v->cur, 0, sizeof v->cur);
            else any = 1;
        }
        const float t = (float)v->frac, gain = vol0 + dvol * (float)(i + 1);
        for (int ch = 0; ch < v->channels; ch++) out[ch][i] = (v->prev[ch] + (v->cur[ch] - v->prev[ch]) * t) * gain;
    }
    v->ramp_volume = v->volume;
    (void)any;
    /* the biquad filters (transposed direct form II, as the DSP's) */
    for (int f = 0; f < 2; f++) {
        if (!v->bq_on[f]) continue;
        const float b0 = v->bq_b[f][0], b1 = v->bq_b[f][1], b2 = v->bq_b[f][2], a1 = v->bq_a[f][0], a2 = v->bq_a[f][1];
        for (int ch = 0; ch < v->channels && ch < MAX_CH; ch++) {
            float z0 = v->bq_z[f][ch][0], z1 = v->bq_z[f][ch][1];
            for (int i = 0; i < FRAME; i++) {
                const float x = out[ch][i], y = b0 * x + z0;
                z0 = b1 * x + a1 * y + z1;   /* (the denominator comes negated: a one-pole low pass is b0 = 1 - p, a1 = p) */
                z1 = b2 * x + a2 * y;
                out[ch][i] = y;
            }
            if (!isfinite(z0) || !isfinite(z1)) {   /* (blown up: silence and start over rather than a full-scale burst) */
                z0 = z1 = 0;
                memset(out[ch], 0, sizeof out[ch]);
            }
            v->bq_z[f][ch][0] = z0;
            v->bq_z[f][ch][1] = z1;
        }
    }
    if (v->dest_kind == DEST_MIX && mixes[v->dest].used) {
        Mix *m = &mixes[v->dest];
        for (int ch = 0; ch < v->channels; ch++)
            for (int b = 0; b < m->buffers; b++) {
                const float g0 = v->ramp_vol[ch][b], g1 = v->vol[ch][b];
                if (g0 == 0.0f && g1 == 0.0f) continue;
                const float dg = (g1 - g0) / FRAME;
                for (int i = 0; i < FRAME; i++) m->buf[b][i] += out[ch][i] * (g0 + dg * (float)(i + 1));
            }
    } else if (v->dest_kind == DEST_SPLITTER && splitters[v->dest].used) {
        Splitter *s = &splitters[v->dest];
        for (int d = 0; d < s->dests; d++) {
            const int mi = s->dest_mix[d];
            if (mi < 0 || !mixes[mi].used) continue;
            Mix *m = &mixes[mi];
            for (int ch = 0; ch < v->channels && ch < MAX_CH; ch++)
                for (int b = 0; b < m->buffers; b++) {
                    const float g0 = s->ramp[d][ch][b], g1 = s->vol[d][ch][b];
                    if (g0 == 0.0f && g1 == 0.0f) continue;
                    const float dg = (g1 - g0) / FRAME;
                    for (int i = 0; i < FRAME; i++) m->buf[b][i] += out[ch][i] * (g0 + dg * (float)(i + 1));
                }
        }
    }
    memcpy(v->ramp_vol, v->vol, sizeof v->ramp_vol);   /* (splitters catch up in render_frame) */
}

static int mix_depth(int i, int guard) {   /* how many sub mixes lie between this one and the final mix */
    if (guard > MAX_MIXES || mixes[i].final_mix || mixes[i].dest_kind != DEST_MIX) return 0;
    return 1 + mix_depth(mixes[i].dest, guard + 1);
}

static void render_frame(int16_t *pcm) {   /* 240 stereo frames */
    for (int m = 0; m < MAX_MIXES; m++)
        if (mixes[m].used) {
            const int rows = mixes[m].buffers < 0 ? 0 : mixes[m].buffers > MAX_BUF ? MAX_BUF : mixes[m].buffers;
            memset(mixes[m].buf, 0, (size_t)rows * sizeof mixes[m].buf[0]);
        }
    for (int i = 0; i < MAX_VOICES; i++)
        if (voices[i].used) render_voice(&voices[i]);
    for (int i = 0; i < MAX_SPLITTERS; i++)   /* (the splitters' volume ramps: once a frame, whichever voices use them) */
        if (splitters[i].used)
            for (int d = 0; d < splitters[i].dests; d++)
                memcpy(splitters[i].ramp[d], splitters[i].vol[d], sizeof splitters[i].ramp[d]);
    /* sub mixes, deepest first */
    int order[MAX_MIXES], n = 0;
    for (int m = 0; m < MAX_MIXES; m++)
        if (mixes[m].used && !mixes[m].final_mix) { mixes[m].depth = mix_depth(m, 0); order[n++] = m; }
    for (int a = 0; a < n; a++)
        for (int b = a + 1; b < n; b++)
            if (mixes[order[b]].depth > mixes[order[a]].depth) { const int t = order[a]; order[a] = order[b]; order[b] = t; }
    for (int k = 0; k < n; k++) {
        Mix *s = &mixes[order[k]];
        effects_frame(order[k]);
        if (s->dest_kind != DEST_MIX || !mixes[s->dest].used) continue;
        Mix *d = &mixes[s->dest];
        for (int src = 0; src < s->buffers; src++)
            for (int dst = 0; dst < d->buffers; dst++) {
                const float g = s->to[src][dst] * s->volume;
                if (g == 0.0f) continue;
                for (int i = 0; i < FRAME; i++) d->buf[dst][i] += s->buf[src][i] * g;
            }
    }
    /* the final mix through the sink's channel map: FL FR FC LFE RL RR, down to stereo */
    float l[FRAME] = {0}, r[FRAME] = {0};
    if (sink_mix >= 0 && mixes[sink_mix].used) {
        effects_frame(sink_mix);
        Mix *f = &mixes[sink_mix];
        const float g = f->volume;
        for (int k = 0; k < sink_n && k < 6; k++) {
            const int b = sink_in[k];
            if (b < 0 || b >= f->buffers) continue;
            const float gl = k == 0 ? 1.0f : k == 1 ? 0.0f : k == 2 ? 0.7071f : k == 4 ? 0.7071f : k == 3 ? 0.5f : 0.0f;
            const float gr = k == 1 ? 1.0f : k == 0 ? 0.0f : k == 2 ? 0.7071f : k == 5 ? 0.7071f : k == 3 ? 0.5f : 0.0f;
            for (int i = 0; i < FRAME; i++) {
                l[i] += f->buf[b][i] * g * gl;
                r[i] += f->buf[b][i] * g * gr;
            }
        }
    }
    {   /* HOS_AUDIO_STATS=1: once a second, what plays and where the signal is */
        static int stats = -1, frames;
        if (stats < 0) stats = getenv("HOS_AUDIO_STATS") != NULL;
        if (stats && ++frames % 200 == 0) {
            int started = 0, with_buffers = 0, to_mix = 0, to_split = 0;
            for (int i = 0; i < MAX_VOICES; i++) {
                const Voice *v = &voices[i];
                if (!v->used) continue;
                started += v->state == PLAY_STARTED;
                with_buffers += v->state == PLAY_STARTED && v->nwb > 0;
                to_mix += v->state == PLAY_STARTED && v->dest_kind == DEST_MIX;
                to_split += v->state == PLAY_STARTED && v->dest_kind == DEST_SPLITTER;
            }
            char b[512];
            int n = snprintf(b, sizeof b, "audio: %d voices started (%d with buffers, %d to a mix, %d to a splitter); mixes:",
                             started, with_buffers, to_mix, to_split);
            for (int m = 0; m < MAX_MIXES && n < 480; m++) {
                if (!mixes[m].used) continue;
                double e = 0;
                for (int k = 0; k < mixes[m].buffers; k++)
                    for (int i = 0; i < FRAME; i++) e += mixes[m].buf[k][i] * mixes[m].buf[k][i];
                n += snprintf(b + n, sizeof b - n, " %d%s(%d->%d v%.2f e%.3g)", m, mixes[m].final_mix ? "F" : "",
                              mixes[m].buffers, mixes[m].dest_kind == DEST_MIX ? mixes[m].dest : -1, mixes[m].volume, e);
            }
            hos_log("%s", b);
            {   /* rate x pitch of the playing voices */
                char h[512];
                int hn = snprintf(h, sizeof h, "audio:   rate/pitch:");
                for (int i = 0; i < MAX_VOICES && hn < 480; i++)
                    if (voices[i].used && voices[i].state == PLAY_STARTED)
                        hn += snprintf(h + hn, sizeof h - hn, " %d/%.3f", voices[i].rate, voices[i].pitch);
                hos_log("%s", h);
            }
            int shown = 0;
            for (int i = 0; i < MAX_VOICES && shown < 4; i++) {
                const Voice *v = &voices[i];
                if (!v->used || v->state != PLAY_STARTED) continue;
                shown++;
                int nz = 0;
                float maxv = 0;
                for (int ch = 0; ch < MAX_CH; ch++)
                    for (int k = 0; k < MAX_BUF; k++)
                        if (v->vol[ch][k] != 0) { nz++; if (fabsf(v->vol[ch][k]) > maxv) maxv = fabsf(v->vol[ch][k]); }
                if (v->dest_kind == DEST_SPLITTER && v->dest >= 0) {
                    const Splitter *sp = &splitters[v->dest];
                    for (int d = 0; d < sp->dests; d++)
                        for (int ch = 0; ch < MAX_CH; ch++)
                            for (int k = 0; k < MAX_BUF; k++)
                                if (sp->vol[d][ch][k] != 0) { nz++; if (fabsf(sp->vol[d][ch][k]) > maxv) maxv = fabsf(sp->vol[d][ch][k]); }
                }
                hos_log("audio:   voice %d fmt %d %d Hz pitch %.2f vol %.2f dest %d/%d, %d mix volumes (max %.2f), %d buffers, pos %lld, played %lld, cur %.4f",
                        i, v->format, v->rate, v->pitch, v->volume, v->dest_kind, v->dest, nz, maxv, v->nwb,
                        (long long)v->pos, (long long)v->played, v->cur[0]);
            }
        }
    }
    if (lim_enabled)   /* fast attack, ~50 ms release */
        for (int i = 0; i < FRAME; i++) {
            const float x = fmaxf(fabsf(l[i]), fabsf(r[i])) * lim_in;
            const float want = x > lim_threshold ? lim_threshold / x : 1.0f;
            lim_env = want < lim_env ? want : lim_env + (want - lim_env) * 0.0004f;
            l[i] *= lim_in * lim_env * lim_out;
            r[i] *= lim_in * lim_env * lim_out;
        }
    for (int i = 0; i < FRAME; i++) {
        float a = l[i] * 32767.0f, b = r[i] * 32767.0f;
        if (a != a) a = 0;   /* (NaN) */
        if (b != b) b = 0;
        if (a > 32767) a = 32767;
        if (a < -32768) a = -32768;
        if (b > 32767) b = 32767;
        if (b < -32768) b = -32768;
        pcm[i * 2] = (int16_t)a;
        pcm[i * 2 + 1] = (int16_t)b;
    }
}

/* ---- output: waveOut, 10 ms blocks; a WAV file with HOS_AUDIO_DUMP ---------------------------------------------- */
#define NBLOCKS 10
static HWAVEOUT wout;
static WAVEHDR hdr[NBLOCKS];
static int16_t blocks[NBLOCKS][FRAME * 2 * 2];
static int fill_block, fill_frames;
static FILE *dump;
static uint32_t dump_bytes;

static void output_open(void) {
    const char *d = getenv("HOS_AUDIO_DUMP");
    if (d && (dump = fopen(d, "wb")) != NULL) {
        static const uint8_t z[44];
        fwrite(z, 1, 44, dump);
    }
    if (getenv("HOS_AUDIO") && atoi(getenv("HOS_AUDIO")) == 0) return;
    WAVEFORMATEX f = {0};
    f.wFormatTag = WAVE_FORMAT_PCM;
    f.nChannels = 2;
    f.nSamplesPerSec = RATE;
    f.wBitsPerSample = 16;
    f.nBlockAlign = 4;
    f.nAvgBytesPerSec = RATE * 4;
    if (waveOutOpen(&wout, WAVE_MAPPER, &f, 0, 0, CALLBACK_NULL) != MMSYSERR_NOERROR) {
        hos_log("audio: no output device");
        wout = NULL;
        return;
    }
    for (int i = 0; i < NBLOCKS; i++) {
        hdr[i].lpData = (LPSTR)blocks[i];
        hdr[i].dwBufferLength = sizeof blocks[i];
        hdr[i].dwFlags = WHDR_DONE;
    }
}
static void dump_finish(void) {
    if (!dump) return;
    uint8_t h[44];
    memcpy(h, "RIFF", 4);
    const uint32_t riff = 36 + dump_bytes, fmt = 16, rate = RATE, brate = RATE * 4, data = dump_bytes;
    const uint16_t pcm = 1, ch = 2, align = 4, bits = 16;
    memcpy(h + 4, &riff, 4);
    memcpy(h + 8, "WAVEfmt ", 8);
    memcpy(h + 16, &fmt, 4);
    memcpy(h + 20, &pcm, 2);
    memcpy(h + 22, &ch, 2);
    memcpy(h + 24, &rate, 4);
    memcpy(h + 28, &brate, 4);
    memcpy(h + 32, &align, 2);
    memcpy(h + 34, &bits, 2);
    memcpy(h + 36, "data", 4);
    memcpy(h + 40, &data, 4);
    fseek(dump, 0, SEEK_SET);
    fwrite(h, 1, 44, dump);
    fflush(dump);
}
static void output_frame(const int16_t *pcm) {
    if (dump) {
        fwrite(pcm, 4, FRAME, dump);
        dump_bytes += FRAME * 4;
        static int n;
        if (++n % 200 == 0) {   /* (the header kept valid every second: the process may end without notice) */
            const long at = ftell(dump);
            dump_finish();
            fseek(dump, at, SEEK_SET);
        }
    }
    if (!wout) return;
    WAVEHDR *h = &hdr[fill_block];
    if (fill_frames == 0) {
        if (!(h->dwFlags & WHDR_DONE)) return;   /* the device is behind: this frame is dropped */
        if (h->dwFlags & WHDR_PREPARED) waveOutUnprepareHeader(wout, h, sizeof *h);
    }
    memcpy(blocks[fill_block] + fill_frames * FRAME * 2, pcm, FRAME * 4);
    if (++fill_frames == 2) {
        h->dwFlags = 0;
        waveOutPrepareHeader(wout, h, sizeof *h);
        waveOutWrite(wout, h, sizeof *h);
        fill_frames = 0;
        fill_block = (fill_block + 1) % NBLOCKS;
    }
}

/* ---- parameter updates ------------------------------------------------------------------------------------------
   The game changes a voice's play state, wave buffers, volumes, filters and destination (and its mixes' and
   splitters') with separate calls, then commits them together with RequestUpdateAudioRenderer; the console's renderer
   only ever sees whole updates. Applied as each call came, the mixer thread rendered frames between them: a voice
   started before its volumes were set (1.0, or the previous sound's), or still sent to its old mix, played a short
   loud blip. The calls queue their change instead (with the arguments copied: TOTK refills the structs it passes),
   and the update applies the queue under the mixer's lock. Should no update come for 50 ms, the mixer applies the
   queue itself. Acquisitions stay immediate (they return the object); releases are queued like the rest, the object
   invalid to the game at once. Objects are queued as our slot indices, looked up when the game calls: the game reuses
   its VoiceType for the next sound right after releasing it, so its handle names the new voice by the time the
   update comes (looked up then, a queued release ended the new sound: no footsteps or hits). A released slot stays
   taken until the update applies the release, so the indices hold for the queue's life.
   HOS_AUDIO_IMMEDIATE=1: every change applied as it comes (as before). */
enum { OP_VOICE_RELEASE, OP_VOICE_DEST, OP_VOICE_MIX_VOLUME, OP_VOICE_VOLUME, OP_VOICE_PITCH, OP_VOICE_BIQUAD,
       OP_VOICE_STATE, OP_VOICE_APPEND, OP_MIX_RELEASE, OP_MIX_VOLUME, OP_MIX_DEST, OP_MIX_MIX_VOLUME, OP_SPLIT_RELEASE,
       OP_SPLIT_DEST, OP_SPLIT_VOLUME };
typedef struct {
    int op, idx, tgt;          /* our voice / mix / splitter; a destination mix or splitter (-1: none) */
    int i0, i1, i2;            /* (op-specific: a kind, an index, channels, a state) */
    uint64_t wbp;              /* an appended WaveBuffer's identity */
    float f, b[3], a[2];
    WB wb;
} Cmd;
static Cmd *cmds;
static int ncmds, cmds_cap;
static int64_t last_update_ns;

static void apply(const Cmd *k) {   /* (with the lock held) */
    if (k->op <= OP_VOICE_APPEND) {
        Voice *v = &voices[k->idx];
        if (!v->used) return;
        switch (k->op) {
        case OP_VOICE_RELEASE: v->used = 0; break;
        case OP_VOICE_DEST:
            v->dest = k->tgt;
            v->dest_kind = k->tgt < 0 ? DEST_NONE
                         : k->i0 == DEST_MIX ? (mixes[k->tgt].used ? DEST_MIX : DEST_NONE)
                         : k->i0 == DEST_SPLITTER ? (splitters[k->tgt].used ? DEST_SPLITTER : DEST_NONE) : DEST_NONE;
            v->snap = 1;
            break;
        case OP_VOICE_MIX_VOLUME: v->vol[k->i0][k->i1] = k->f; break;
        case OP_VOICE_VOLUME: v->volume = k->f; break;
        case OP_VOICE_PITCH: v->pitch = k->f; break;
        case OP_VOICE_BIQUAD:
            if (k->i1 && !v->bq_on[k->i0]) memset(v->bq_z[k->i0], 0, sizeof v->bq_z[k->i0]);
            v->bq_on[k->i0] = k->i1;
            memcpy(v->bq_b[k->i0], k->b, sizeof k->b);
            memcpy(v->bq_a[k->i0], k->a, sizeof k->a);
            break;
        case OP_VOICE_STATE:
            if (k->i0 == PLAY_STOPPED && v->state != PLAY_STOPPED) {   /* stopped: every queued buffer is released */
                for (int i = 0; i < v->nwb; i++) release_wb(v, v->wb[i]);
                v->nwb = 0;
                v->started_wb = 0;
                v->frac = 0;
                memset(v->prev, 0, sizeof v->prev);
                memset(v->cur, 0, sizeof v->cur);
                v->yn1 = v->yn2 = 0;
            }
            if (k->i0 == PLAY_STARTED && v->state != PLAY_STARTED) v->snap = 1;
            v->state = k->i0;
            break;
        case OP_VOICE_APPEND:
            if (v->pending_wb > 0) v->pending_wb--;
            if (v->nwb < MAX_WB) {
                v->wbc[v->nwb] = k->wb;
                v->wb[v->nwb++] = k->wbp;
            } else {
                release_wb(v, k->wbp);
            }
            break;
        }
        return;
    }
    const int tgt_mix = k->tgt >= 0 && k->tgt < MAX_MIXES && mixes[k->tgt].used ? k->tgt : -1;
    if (k->op <= OP_MIX_MIX_VOLUME) {
        Mix *m = &mixes[k->idx];
        if (!m->used) return;
        switch (k->op) {
        case OP_MIX_RELEASE:
            m->used = 0;
            if (sink_mix == k->idx) sink_mix = -1;
            break;
        case OP_MIX_VOLUME: m->volume = k->f; break;
        case OP_MIX_DEST:   /* (a sub mix into a splitter: not used by TOTK so far) */
            m->dest = k->i0 == DEST_MIX ? tgt_mix : -1;
            m->dest_kind = m->dest >= 0 ? DEST_MIX : DEST_NONE;
            break;
        case OP_MIX_MIX_VOLUME: m->to[k->i0][k->i1] = k->f; break;
        }
        return;
    }
    Splitter *s = &splitters[k->idx];
    if (!s->used) return;
    switch (k->op) {
    case OP_SPLIT_RELEASE: s->used = 0; break;
    case OP_SPLIT_DEST: s->dest_mix[k->i0] = tgt_mix; break;
    case OP_SPLIT_VOLUME:
        s->vol[k->i0][k->i1][k->i2] = k->f;
        if (s->dest_mix[k->i0] < 0) s->dest_mix[k->i0] = tgt_mix;
        break;
    }
}
static void submit(const Cmd *k) {   /* (with the lock held) */
    static int immediate = -1;
    if (immediate < 0) immediate = getenv("HOS_AUDIO_IMMEDIATE") != NULL;
    if (ncmds == cmds_cap && !immediate) {
        const int cap = cmds_cap ? cmds_cap * 2 : 4096;
        Cmd *grown = (Cmd *)realloc(cmds, (size_t)cap * sizeof *grown);
        if (grown) cmds = grown, cmds_cap = cap;
    }
    if (immediate || ncmds == cmds_cap) apply(k);
    else cmds[ncmds++] = *k;
}
static void commit(void) {   /* (with the lock held) */
    for (int i = 0; i < ncmds; i++) apply(&cmds[i]);
    ncmds = 0;
}

/* ---- the renderer thread --------------------------------------------------------------------------------------- */
static uint64_t renderer_event;
static volatile int renderer_running;
static volatile int64_t elapsed_frames;
static HANDLE renderer_thread;

static DWORD WINAPI renderer_main(void *p) {
    (void)p;
    hos_thread_boost(2);
    output_open();
    int64_t period = 5000000, next = hos_now_ns() + period;   /* one audio frame: 240 samples at 48 kHz */
    static int16_t pcm[FRAME * 2];
    for (;;) {
        hos_sleep_ns(next - hos_now_ns());
        next += period;
        if (hos_now_ns() - next > 100000000) next = hos_now_ns() + period;   /* (fell far behind: catch up) */
        if (!renderer_running) continue;
        au_lock();
        if (ncmds && hos_now_ns() - last_update_ns > 50000000) commit();   /* (no update came: see "parameter updates") */
        render_frame(pcm);
        au_unlock();
        output_frame(pcm);
        elapsed_frames++;
        if (renderer_event) hos_system_event_signal(renderer_event);
    }
}

/* ---- renderer lifecycle ---------------------------------------------------------------------------------------- */
/* OpenAudioRenderer(AudioRendererHandle*, os::SystemEvent*, AudioRendererParameter const&, void*, unsigned long) */
HLE(_ZN2nn5audio17OpenAudioRendererEPNS0_19AudioRendererHandleEPNS_2os11SystemEventERKNS0_22AudioRendererParameterEPvm) {
    G64(ARG(0)) = 1;
    /* nn::os::SystemEvent (the C++ class) starts with its SystemEventType: key the event on that address */
    renderer_event = ARG(1);
    hos_system_event_init(renderer_event, 1);
    au_lock();
    au_unlock();
    if (!renderer_thread) renderer_thread = CreateThread(NULL, 0, renderer_main, NULL, 0, NULL);
    hos_log("audio: renderer opened");
    RET(0);
}
HLE(_ZN2nn5audio18StartAudioRendererENS0_19AudioRendererHandleE) { renderer_running = 1; RET(0); }
HLE(_ZN2nn5audio17StopAudioRendererENS0_19AudioRendererHandleE) { renderer_running = 0; }
HLE(_ZN2nn5audio18CloseAudioRendererENS0_19AudioRendererHandleE) { renderer_running = 0; renderer_event = 0; dump_finish(); }
HLE(_ZN2nn5audio26RequestUpdateAudioRendererENS0_19AudioRendererHandleEPKNS0_19AudioRendererConfigE) {
    au_lock();
    commit();
    last_update_ns = hos_now_ns();
    au_unlock();
    RET(0);
}
HLE(_ZN2nn5audio33GetAudioRendererElapsedFrameCountEPKNS0_19AudioRendererConfigE) { RET(elapsed_frames); }

/* Sizes the game allocates for the renderer: generous, never touched by us. */
HLE(_ZN2nn5audio30GetAudioRendererWorkBufferSizeERKNS0_22AudioRendererParameterE) { RET(0x10000); }
HLE(_ZN2nn5audio36GetAudioRendererConfigWorkBufferSizeERKNS0_22AudioRendererParameterE) { RET(0x10000); }
HLE(_ZN2nn5audio41GetRequiredBufferSizeForPerformanceFramesERKNS0_22AudioRendererParameterE) { RET(0x1000); }
HLE(_ZN2nn5audio35GetRequiredBufferSizeForI3dl2ReverbEii) { RET(0x10000); }
HLE(_ZN2nn5audio30GetRequiredBufferSizeForReverbEii) { RET(0x10000); }
HLE(_ZN2nn5audio29GetRequiredBufferSizeForDelayENS_8TimeSpanEii) { RET(0x10000); }
HLE(_ZN2nn5audio36GetRequiredBufferSizeForLightLimiterENS_8TimeSpanEii) { RET(0x1000); }
HLE(_ZN2nn5audio43GetRequiredBufferSizeForAuxSendReturnBufferEPKNS0_22AudioRendererParameterEii) {
    RET((uint64_t)(int)ARG(1) * (uint64_t)(int)ARG(2) * 4 + 0x100);   /* frames x channels of int32 */
}

/* AudioRendererParameter: sampleRate, sampleCount, mixBufferCount, subMixCount, voiceCount, sinkCount, effectCount,
   performanceFrameCount, isVoiceDropEnabled, splitterCount, splitterSendChannelCount, renderingDevice,
   executionMode, magic: the defaults; the game sets its own values after. */
HLE(_ZN2nn5audio32InitializeAudioRendererParameterEPNS0_22AudioRendererParameterE) {
    uint32_t *p = (uint32_t *)GP(ARG(0));
    p[0] = RATE;
    p[1] = FRAME;
    p[2] = 1;
    p[3] = 0;
    p[4] = 24;
    p[5] = 1;
    p[6] = 0;
    p[7] = 0;
}
HLE(_ZN2nn5audio29InitializeAudioRendererConfigEPNS0_19AudioRendererConfigERKNS0_22AudioRendererParameterEPvm) {
    put_handle(ARG(0), K_CONFIG, 0);
}

/* The output device: one stereo device, "MainAudioOut". AudioDeviceName is char[256]. */
HLE(_ZN2nn5audio39AcquireAudioDeviceNotificationForOutputEPNS_2os11SystemEventE) { hos_system_event_init(ARG(0), 1); }
HLE(_ZN2nn5audio41GetActiveAudioDeviceChannelCountForOutputEv) { RET(2); }
HLE(_ZN2nn5audio30GetActiveAudioOutputDeviceNameEPNS0_15AudioDeviceNameE) {
    memset(GP(ARG(0)), 0, 256);
    memcpy(GP(ARG(0)), "MainAudioOut", 12);
}
HLE(_ZN2nn5audio25ListAudioOutputDeviceNameEPNS0_15AudioDeviceNameEi) {
    if ((int)ARG(1) > 0) {
        memset(GP(ARG(0)), 0, 256);
        memcpy(GP(ARG(0)), "MainAudioOut", 12);
    }
    RET((int)ARG(1) > 0 ? 1 : 0);
}
HLE(_ZN2nn5audio26SetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameEf) {}

/* GetSampleByteSize(SampleFormat): PcmInt8 1, PcmInt16 2, PcmInt24 3, PcmInt32 4, PcmFloat 4 (ADPCM: no fixed size).
   The game steps through sample buffers by it: 0 kept the main thread looping (f_089d3588) while Great Sky Island
   loaded. */
HLE(_ZN2nn5audio17GetSampleByteSizeENS0_12SampleFormatE) {
    static const uint8_t size[7] = {0, 1, 2, 3, 4, 4, 0};
    const uint32_t f = (uint32_t)ARG(0);
    RET(f < 7 ? size[f] : 0);
}

/* ---- memory pools: bookkeeping only (all guest memory is readable) --------------------------------------------- */
HLE(_ZN2nn5audio17AcquireMemoryPoolEPNS0_19AudioRendererConfigEPNS0_14MemoryPoolTypeEPvm) {
    uint64_t *p = (uint64_t *)GP(ARG(1));
    p[0] = (uint64_t)AU_MAGIC << 32 | (uint64_t)K_POOL << 24;
    p[1] = ARG(2);
    p[2] = ARG(3);
    RET(1);
}
HLE(_ZN2nn5audio17ReleaseMemoryPoolEPNS0_19AudioRendererConfigEPNS0_14MemoryPoolTypeE) { G64(ARG(1)) = 0; }
HLE(_ZN2nn5audio23RequestAttachMemoryPoolEPNS0_14MemoryPoolTypeE) { RET(1); }
HLE(_ZN2nn5audio23RequestDetachMemoryPoolEPNS0_14MemoryPoolTypeE) { RET(1); }
HLE(_ZN2nn5audio20IsMemoryPoolAttachedEPKNS0_14MemoryPoolTypeE) { RET(1); }
HLE(_ZN2nn5audio20GetMemoryPoolAddressEPKNS0_14MemoryPoolTypeE) { RET(G64(ARG(0) + 8)); }
HLE(_ZN2nn5audio17GetMemoryPoolSizeEPKNS0_14MemoryPoolTypeE) { RET(G64(ARG(0) + 16)); }

/* ---- mixes ------------------------------------------------------------------------------------------------------ */
static int new_mix(uint64_t obj, int final_mix, int buffers) {
    au_lock();
    int i = 0;
    while (i < MAX_MIXES && mixes[i].used) i++;
    if (i == MAX_MIXES) { au_unlock(); return 0; }
    memset(&mixes[i], 0, sizeof mixes[i]);
    mixes[i].used = 1;
    mixes[i].final_mix = final_mix;
    mixes[i].buffers = buffers < 1 ? 1 : buffers > MAX_BUF ? MAX_BUF : buffers;
    mixes[i].volume = 1.0f;
    put_handle(obj, K_MIX, i);
    AUT("audio: %s mix %d (%d buffers) at %#llx", final_mix ? "final" : "sub", i, buffers, (unsigned long long)obj);
    au_unlock();
    return 1;
}
/* AcquireFinalMix(config, FinalMixType*, int bufferCount) */
HLE(_ZN2nn5audio15AcquireFinalMixEPNS0_19AudioRendererConfigEPNS0_12FinalMixTypeEi) { RET(new_mix(ARG(1), 1, (int)ARG(2))); }
/* AcquireSubMix(config, SubMixType*, int sampleRate, int bufferCount) */
HLE(_ZN2nn5audio13AcquireSubMixEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeEii) { RET(new_mix(ARG(1), 0, (int)ARG(3))); }
static void release_mix(uint64_t obj) {
    au_lock();
    Mix *m = mix_of(obj);
    if (m) {
        m->released = 1;
        const Cmd k = {.op = OP_MIX_RELEASE, .idx = (int)(m - mixes)};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio15ReleaseFinalMixEPNS0_19AudioRendererConfigEPNS0_12FinalMixTypeE) { release_mix(ARG(1)); }
HLE(_ZN2nn5audio13ReleaseSubMixEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeE) { release_mix(ARG(1)); }
static void set_mix_volume(Ctx *c) {
    au_lock();
    if (mix_of(ARG(0))) {
        const Cmd k = {.op = OP_MIX_VOLUME, .idx = mix_index(ARG(0)), .f = farg(c, 0)};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio17SetFinalMixVolumeEPNS0_12FinalMixTypeEf) { set_mix_volume(c); }
HLE(_ZN2nn5audio15SetSubMixVolumeEPNS0_10SubMixTypeEf) {
    AUT("audio: sub mix %d volume %.3f", mix_index(ARG(0)), farg(c, 0));
    set_mix_volume(c);
}
static void set_mix_dest(uint64_t obj, int kind, uint64_t dest) {
    au_lock();
    AUT("audio: sub mix %d destination %s %#llx (mix %d, splitter %d)", mix_index(obj), kind == DEST_MIX ? "mix" : "splitter",
        (unsigned long long)dest, mix_index(dest), get_handle(dest, K_SPLITTER));
    if (mix_of(obj)) {
        const Cmd k = {.op = OP_MIX_DEST, .idx = mix_index(obj), .tgt = kind == DEST_MIX ? mix_index(dest) : -1, .i0 = kind};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio20SetSubMixDestinationEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeEPNS0_12FinalMixTypeE) { set_mix_dest(ARG(1), DEST_MIX, ARG(2)); }
HLE(_ZN2nn5audio20SetSubMixDestinationEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeES4_) { set_mix_dest(ARG(1), DEST_MIX, ARG(2)); }
HLE(_ZN2nn5audio20SetSubMixDestinationEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeEPNS0_12SplitterTypeE) {
    set_mix_dest(ARG(1), DEST_SPLITTER, ARG(2));
}
HLE(_ZN2nn5audio22ClearSubMixDestinationEPNS0_19AudioRendererConfigEPNS0_10SubMixTypeE) { set_mix_dest(ARG(1), DEST_MIX, 0); }
/* SetSubMixMixVolume(SubMixType*, destination, float volume, int src, int dst) */
static void set_submix_volume(Ctx *c) {
    au_lock();
    const int src = (int)ARG(2), dst = (int)ARG(3);
    if (mix_of(ARG(0)) && src >= 0 && src < MAX_BUF && dst >= 0 && dst < MAX_BUF) {
        const Cmd k = {.op = OP_MIX_MIX_VOLUME, .idx = mix_index(ARG(0)), .i0 = src, .i1 = dst, .f = farg(c, 0)};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio18SetSubMixMixVolumeEPNS0_10SubMixTypeEPNS0_12FinalMixTypeEfii) { set_submix_volume(c); }
HLE(_ZN2nn5audio18SetSubMixMixVolumeEPNS0_10SubMixTypeES2_fii) { set_submix_volume(c); }

/* ---- splitters -------------------------------------------------------------------------------------------------- */
/* AcquireSplitter(config, SplitterType*, int sampleRate, int sourceChannelCount, int destinationCount) */
HLE(_ZN2nn5audio15AcquireSplitterEPNS0_19AudioRendererConfigEPNS0_12SplitterTypeEiii) {
    au_lock();
    int i = 0;
    while (i < MAX_SPLITTERS && splitters[i].used) i++;
    if (i == MAX_SPLITTERS) { au_unlock(); RET(0); return; }
    memset(&splitters[i], 0, sizeof splitters[i]);
    splitters[i].used = 1;
    splitters[i].channels = (int)ARG(3);
    splitters[i].dests = (int)ARG(4) > MAX_SPLIT_DESTS ? MAX_SPLIT_DESTS : (int)ARG(4);
    for (int d = 0; d < MAX_SPLIT_DESTS; d++) splitters[i].dest_mix[d] = -1;
    put_handle(ARG(1), K_SPLITTER, i);
    au_unlock();
    RET(1);
}
HLE(_ZN2nn5audio15ReleaseSplitterEPNS0_19AudioRendererConfigEPNS0_12SplitterTypeE) {
    au_lock();
    Splitter *s = splitter_of(ARG(1));
    if (s) {
        s->released = 1;
        const Cmd k = {.op = OP_SPLIT_RELEASE, .idx = (int)(s - splitters)};
        submit(&k);
    }
    au_unlock();
}
/* SetSplitterDestination(config, SplitterType*, int destinationIndex, mix) */
static void set_split_dest(Ctx *c) {
    au_lock();
    const int d = (int)ARG(2);
    if (splitter_of(ARG(1)) && d >= 0 && d < MAX_SPLIT_DESTS) {
        const Cmd k = {.op = OP_SPLIT_DEST, .idx = splitter_index(ARG(1)), .tgt = mix_index(ARG(3)), .i0 = d};
        submit(&k);
    }
    AUT("audio: splitter %d destination %d = mix %d", get_handle(ARG(1), K_SPLITTER), d, mix_index(ARG(3)));
    au_unlock();
}
HLE(_ZN2nn5audio22SetSplitterDestinationEPNS0_19AudioRendererConfigEPNS0_12SplitterTypeEiPNS0_10SubMixTypeE) { set_split_dest(c); }
HLE(_ZN2nn5audio22SetSplitterDestinationEPNS0_19AudioRendererConfigEPNS0_12SplitterTypeEiPNS0_12FinalMixTypeE) { set_split_dest(c); }
/* SetSplitterMixVolume(SplitterType*, int destinationIndex, mix, float volume, int src, int dst) */
static void set_split_volume(Ctx *c) {
    au_lock();
    const int d = (int)ARG(1), src = (int)ARG(3), dst = (int)ARG(4);
    if (splitter_of(ARG(0)) && d >= 0 && d < MAX_SPLIT_DESTS && src >= 0 && src < MAX_CH && dst >= 0 && dst < MAX_BUF) {
        const Cmd k = {.op = OP_SPLIT_VOLUME, .idx = splitter_index(ARG(0)), .tgt = mix_index(ARG(2)), .i0 = d, .i1 = src,
                       .i2 = dst, .f = farg(c, 0)};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio20SetSplitterMixVolumeEPNS0_12SplitterTypeEiPNS0_10SubMixTypeEfii) { set_split_volume(c); }
HLE(_ZN2nn5audio20SetSplitterMixVolumeEPNS0_12SplitterTypeEiPNS0_12FinalMixTypeEfii) { set_split_volume(c); }

/* ---- the device sink: AddDeviceSink(config, DeviceSinkType*, FinalMixType*, const int8* input, int count, name) - */
HLE(_ZN2nn5audio13AddDeviceSinkEPNS0_19AudioRendererConfigEPNS0_14DeviceSinkTypeEPNS0_12FinalMixTypeEPKaiPKc) {
    au_lock();
    sink_mix = mix_index(ARG(2));
    sink_n = (int)ARG(4) > 8 ? 8 : (int)ARG(4);
    memcpy(sink_in, GP(ARG(3)), (size_t)(sink_n > 0 ? sink_n : 0));
    put_handle(ARG(1), K_SINK, 0);
    hos_log("audio: device sink on mix %d, %d channels (%d %d %d %d %d %d)", sink_mix, sink_n, sink_in[0], sink_in[1],
            sink_in[2], sink_in[3], sink_in[4], sink_in[5]);
    au_unlock();
    RET(0);
}
HLE(_ZN2nn5audio16RemoveDeviceSinkEPNS0_19AudioRendererConfigEPNS0_14DeviceSinkTypeEPNS0_12FinalMixTypeE) {
    au_lock();
    sink_mix = -1;
    au_unlock();
}

/* ---- voices ----------------------------------------------------------------------------------------------------- */
/* AcquireVoiceSlot(config, VoiceType*, int sampleRate, int channelCount, SampleFormat, int priority,
                    const void* parameter, size_t parameterSize, const BehaviorOptions*) */
HLE(_ZN2nn5audio16AcquireVoiceSlotEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeEiiNS0_12SampleFormatEiPKvmPKNS3_15BehaviorOptionsE) {
    au_lock();
    int i = 0;
    while (i < MAX_VOICES && voices[i].used) i++;
    if (i == MAX_VOICES) { au_unlock(); RET(0); return; }
    Voice *v = &voices[i];
    memset(v, 0, sizeof *v);
    v->used = 1;
    v->guest = ARG(1);
    v->rate = (int)ARG(2) > 0 ? (int)ARG(2) : RATE;
    v->channels = (int)ARG(3) < 1 ? 1 : (int)ARG(3) > MAX_CH ? MAX_CH : (int)ARG(3);
    v->format = (int)ARG(4);
    v->param = ARG(6);
    if (v->param && ARG(7) >= 32) memcpy(v->coefs, GP(v->param), 32);
    v->volume = 1.0f;
    v->pitch = 1.0f;
    v->state = PLAY_STOPPED;
    put_handle(ARG(1), K_VOICE, i);
    static int logged;
    if (logged++ < 12 || au_trace > 0)
        hos_log("audio: voice %d: %d Hz, %d channels, format %d, parameter %#llx (%llu bytes) at %.1f s", i, v->rate, v->channels,
                v->format, (unsigned long long)ARG(6), (unsigned long long)ARG(7), hos_now_ns() / 1e9);
    au_unlock();
    RET(1);
}
HLE(_ZN2nn5audio16ReleaseVoiceSlotEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeE) {
    au_lock();
    Voice *v = voice_of(ARG(1));
    if (v) {
        v->released = 1;
        const Cmd k = {.op = OP_VOICE_RELEASE, .idx = (int)(v - voices)};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio12IsVoiceValidEPKNS0_9VoiceTypeE) {
    au_lock();
    RET(voice_of(ARG(0)) != NULL);
    au_unlock();
}
static void set_voice_dest(Ctx *c, int kind) {
    au_lock();
    if (voice_of(ARG(1))) {
        const Cmd k = {.op = OP_VOICE_DEST, .idx = voice_index(ARG(1)), .i0 = kind,
                       .tgt = kind == DEST_MIX ? mix_index(ARG(2)) : splitter_index(ARG(2))};
        submit(&k);
    }
    au_unlock();
    RET(1);
}
HLE(_ZN2nn5audio19SetVoiceDestinationEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeEPNS0_10SubMixTypeE) { set_voice_dest(c, DEST_MIX); }
HLE(_ZN2nn5audio19SetVoiceDestinationEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeEPNS0_12FinalMixTypeE) { set_voice_dest(c, DEST_MIX); }
HLE(_ZN2nn5audio19SetVoiceDestinationEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeEPNS0_12SplitterTypeE) { set_voice_dest(c, DEST_SPLITTER); }
HLE(_ZN2nn5audio21ClearVoiceDestinationEPNS0_19AudioRendererConfigEPNS0_9VoiceTypeE) {
    au_lock();
    if (voice_of(ARG(1))) {
        const Cmd k = {.op = OP_VOICE_DEST, .idx = voice_index(ARG(1)), .tgt = -1, .i0 = DEST_NONE};
        submit(&k);
    }
    au_unlock();
}
/* SetVoiceMixVolume(VoiceType*, mix, float volume, int src, int dst) */
static void set_voice_mix_volume(Ctx *c) {
    au_lock();
    const int src = (int)ARG(2), dst = (int)ARG(3);
    if (voice_of(ARG(0)) && src >= 0 && src < MAX_CH && dst >= 0 && dst < MAX_BUF) {
        const Cmd k = {.op = OP_VOICE_MIX_VOLUME, .idx = voice_index(ARG(0)), .i0 = src, .i1 = dst, .f = farg(c, 0)};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio17SetVoiceMixVolumeEPNS0_9VoiceTypeEPNS0_10SubMixTypeEfii) { set_voice_mix_volume(c); }
HLE(_ZN2nn5audio17SetVoiceMixVolumeEPNS0_9VoiceTypeEPNS0_12FinalMixTypeEfii) { set_voice_mix_volume(c); }
HLE(_ZN2nn5audio14SetVoiceVolumeEPNS0_9VoiceTypeEf) {
    au_lock();
    Voice *v = voice_of(ARG(0));
    if (v) {
        if (v->volume != farg(c, 0)) AUT("audio: voice %d volume %.3f", (int)(v - voices), farg(c, 0));
        const Cmd k = {.op = OP_VOICE_VOLUME, .idx = (int)(v - voices), .f = farg(c, 0)};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio13SetVoicePitchEPNS0_9VoiceTypeEf) {
    au_lock();
    if (voice_of(ARG(0))) {
        const Cmd k = {.op = OP_VOICE_PITCH, .idx = voice_index(ARG(0)), .f = farg(c, 0)};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio16SetVoicePriorityEPNS0_9VoiceTypeEi) {}
/* BiquadFilterParameter: bool enable @0, int16 numerator[3] @2, int16 denominator[2] @8 (Q14) */
HLE(_ZN2nn5audio29SetVoiceBiquadFilterParameterEPNS0_9VoiceTypeEiRKNS0_21BiquadFilterParameterE) {
    au_lock();
    Voice *v = voice_of(ARG(0));
    const int f = (int)ARG(1);
    if (v && f >= 0 && f < 2 && ARG(2)) {
        const uint64_t p = ARG(2);
        Cmd k = {.op = OP_VOICE_BIQUAD, .idx = (int)(v - voices), .i0 = f, .i1 = *(uint8_t *)GP(p) != 0};
        for (int n = 0; n < 3; n++) k.b[n] = *(int16_t *)GP(p + 2 + 2 * n) / 16384.0f;
        for (int n = 0; n < 2; n++) k.a[n] = *(int16_t *)GP(p + 8 + 2 * n) / 16384.0f;
        submit(&k);
        static int logged;
        if (k.i1 && logged++ < 8) AUT("audio: voice biquad %d: b %.3f %.3f %.3f a %.3f %.3f", f, k.b[0], k.b[1], k.b[2], k.a[0], k.a[1]);
    }
    au_unlock();
}
/* SetVoicePlayState(VoiceType*, PlayState): Started 0, Stopped 1, Paused 2 */
HLE(_ZN2nn5audio17SetVoicePlayStateEPNS0_9VoiceTypeENS1_9PlayStateE) {
    au_lock();
    Voice *v = voice_of(ARG(0));
    if (v) {
        const int s = (int)ARG(1);
        if (v->state != s) AUT("audio: voice %d %s", (int)(v - voices), s == 0 ? "start" : s == 1 ? "stop" : "pause");
        const Cmd k = {.op = OP_VOICE_STATE, .idx = (int)(v - voices), .i0 = s};
        submit(&k);
    }
    au_unlock();
}
HLE(_ZN2nn5audio16AppendWaveBufferEPNS0_9VoiceTypeEPKNS0_10WaveBufferE) {
    au_lock();
    Voice *v = voice_of(ARG(0));
    int ok = 0;
    if (v && v->nwb + v->pending_wb < MAX_WB) {
        {   /* HOS_AUDIO_WB_DUMP=<dir>: the first 40 wave buffers' data and the voice's coefficients (debugging) */
            static const char *dir;
            static int init, nd;
            if (!init) { init = 1; dir = getenv("HOS_AUDIO_WB_DUMP"); }
            if (dir && nd < 40) {
                const WB w = read_wb(ARG(1));
                char path[512];
                snprintf(path, sizeof path, "%s/wb%02d_v%d_f%d_r%d_s%d_e%d.bin", dir, nd++, (int)(v - voices), v->format, v->rate,
                         w.start, w.end);
                FILE *f = fopen(path, "wb");
                if (f) {
                    fwrite(v->coefs, 1, 32, f);
                    fwrite(GP(w.buffer), 1, (size_t)w.size, f);
                    fclose(f);
                }
            }
        }
        const Cmd k = {.op = OP_VOICE_APPEND, .idx = (int)(v - voices), .wbp = ARG(1), .wb = read_wb(ARG(1))};
        v->pending_wb++;
        submit(&k);
        ok = 1;
        static int logged;
        if (logged++ < 8) {
            const WB w = read_wb(ARG(1));
            hos_log("audio: wave buffer %#llx: data %#llx size %llu samples %d..%d loop %d context %#llx (voice format %d)",
                    (unsigned long long)ARG(1), (unsigned long long)w.buffer, (unsigned long long)w.size, w.start, w.end,
                    w.loop, (unsigned long long)w.context, v->format);
        }
    }
    au_unlock();
    RET(ok);
}
HLE(_ZN2nn5audio21GetReleasedWaveBufferEPNS0_9VoiceTypeE) {
    au_lock();
    Voice *v = voice_of(ARG(0));
    uint64_t r = 0;
    if (v && v->nrel) {
        r = v->rel[0];
        memmove(v->rel, v->rel + 1, (size_t)(v->nrel - 1) * sizeof v->rel[0]);
        v->nrel--;
    }
    au_unlock();
    RET(r);
}
HLE(_ZN2nn5audio18GetWaveBufferCountEPKNS0_9VoiceTypeE) {
    au_lock();
    Voice *v = voice_of(ARG(0));
    RET(v ? v->nwb + v->pending_wb : 0);
    au_unlock();
}
HLE(_ZN2nn5audio25GetVoicePlayedSampleCountEPKNS0_9VoiceTypeE) {
    au_lock();
    Voice *v = voice_of(ARG(0));
    RET(v ? v->played : 0);
    au_unlock();
}
HLE(_ZN2nn5audio20IsVoiceDroppedFlagOnEPKNS0_9VoiceTypeE) { RET(0); }

/* ---- effects: accepted, not processed (the dry signal goes through) -------------------------------------------- */
HLE(_ZN2nn5audio6AddAuxEPNS0_19AudioRendererConfigEPNS0_7AuxTypeEPNS0_10SubMixTypeEPvS7_m) { RET(0); }
HLE(_ZN2nn5audio6AddAuxEPNS0_19AudioRendererConfigEPNS0_7AuxTypeEPNS0_12FinalMixTypeEPvS7_m) { RET(0); }
HLE(_ZN2nn5audio12IsAuxEnabledEPKNS0_7AuxTypeE) { RET(1); }
HLE(_ZN2nn5audio14IsAuxRemovableEPKNS0_7AuxTypeE) { RET(1); }
HLE(_ZN2nn5audio17ReadAuxSendBufferEPNS0_7AuxTypeEPii) { RET(0); }
HLE(_ZN2nn5audio20WriteAuxReturnBufferEPNS0_7AuxTypeEPKii) { RET((int)ARG(2)); }
HLE(_ZN2nn5audio17GetAuxSampleCountEPKNS0_7AuxTypeE) { RET(0); }
HLE(_ZN2nn5audio17IsReverbRemovableEPKNS0_10ReverbTypeE) { RET(1); }
HLE(_ZN2nn5audio22IsBufferMixerRemovableEPKNS0_15BufferMixerTypeE) { RET(1); }
HLE(_ZN2nn5audio22IsI3dl2ReverbRemovableEPKNS0_15I3dl2ReverbTypeE) { RET(1); }
HLE(_ZN2nn5audio23IsBiquadFilterRemovableEPKNS0_16BiquadFilterTypeE) { RET(1); }
HLE(_ZN2nn5audio23IsLightLimiterRemovableEPKNS0_16LightLimiterTypeE) { RET(1); }

/* ---- the light limiter on the final mix: input gain, then a peak limiter at the threshold, then output gain ---- */
HLE(_ZN2nn5audio24SetLightLimiterInputGainEPNS0_16LightLimiterTypeEf) { lim_in = farg(c, 0); AUT("audio: limiter input gain %.3f", lim_in); }
HLE(_ZN2nn5audio25SetLightLimiterOutputGainEPNS0_16LightLimiterTypeEf) { lim_out = farg(c, 0); AUT("audio: limiter output gain %.3f", lim_out); }
HLE(_ZN2nn5audio24SetLightLimiterThresholdEPNS0_16LightLimiterTypeEf) { lim_threshold = farg(c, 0); AUT("audio: limiter threshold %.3f", lim_threshold); }
HLE(_ZN2nn5audio22SetLightLimiterEnabledEPNS0_16LightLimiterTypeEb) { lim_enabled = (int)(ARG(1) & 1); AUT("audio: limiter %s", lim_enabled ? "on" : "off"); }
HLE(_ZN2nn5audio21IsLightLimiterEnabledEPKNS0_16LightLimiterTypeE) { RET(lim_enabled); }

/* ---- I3DL2 reverb: AddI3dl2Reverb(config, I3dl2ReverbType*, buffer, size, SubMixType* / FinalMixType*, channels) - */
static Reverb *reverb_of(uint64_t obj) {
    const int i = get_handle(obj, K_EFFECT);
    return i >= 0 && i < MAX_EFFECTS && reverbs[i].used ? &reverbs[i] : NULL;
}
static void add_reverb(Ctx *c) {
    au_lock();
    int e = 0;
    while (e < MAX_EFFECTS && reverbs[e].used) e++;
    const int m = mix_index(ARG(4));
    if (e == MAX_EFFECTS || m < 0) {
        au_unlock();
        AUT("audio: i3dl2 reverb refused (mix %d)", m);
        RET(0x1A9 << 9 | 153);   /* (out of effects: an audio result) */
        return;
    }
    Reverb *rv = &reverbs[e];
    memset(rv, 0, sizeof *rv);
    rv->used = 1;
    rv->mix = m;
    rv->n = (int)ARG(5) > MAX_CH ? MAX_CH : (int)ARG(5);
    for (int k = 0; k < MAX_CH; k++) rv->in[k] = rv->out[k] = (int8_t)k;
    /* the I3DL2 defaults */
    rv->room = -1000; rv->room_hf = -100; rv->refl = -2602; rv->reverb = 200; rv->hf_ratio = 0.83f;
    rv->diffusion = 100; rv->density = 100; rv->hf_ref = 5000; rv->dry = 1.0f;
    rv->decay_ns = 1490000000ll; rv->refl_delay_ns = 7000000; rv->late_delay_ns = 11000000;
    put_handle(ARG(1), K_EFFECT, e);
    au_unlock();
    AUT("audio: i3dl2 reverb %d on mix %d, %d channels", e, m, rv->n);
    RET(0);
}
HLE(_ZN2nn5audio14AddI3dl2ReverbEPNS0_19AudioRendererConfigEPNS0_15I3dl2ReverbTypeEPvmPNS0_10SubMixTypeEi) { add_reverb(c); }
HLE(_ZN2nn5audio14AddI3dl2ReverbEPNS0_19AudioRendererConfigEPNS0_15I3dl2ReverbTypeEPvmPNS0_12FinalMixTypeEi) { add_reverb(c); }
static void ret_float(Ctx *c, float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    memset(&c->vreg[0], 0, sizeof c->vreg[0]);
    memcpy(&c->vreg[0][0], &bits, 4);
}
#define REVERB_FLOAT(NAME, field) \
    HLE(_ZN2nn5audio##NAME##EPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->field = farg(c, 0); }
#define REVERB_FLOAT_GET(NAME, field) \
    HLE(_ZN2nn5audio##NAME##EPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->field : 0.0f); }
#define REVERB_TIME(NAME, field) \
    HLE(_ZN2nn5audio##NAME##EPNS0_15I3dl2ReverbTypeENS_8TimeSpanE) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->field = (int64_t)ARG(1); }
#define REVERB_TIME_GET(NAME, field) \
    HLE(_ZN2nn5audio##NAME##EPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); RET(rv ? (uint64_t)rv->field : 0); }
HLE(_ZN2nn5audio22SetI3dl2ReverbRoomGainEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->room = farg(c, 0); }
HLE(_ZN2nn5audio22GetI3dl2ReverbRoomGainEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->room : 0.0f); }
HLE(_ZN2nn5audio24SetI3dl2ReverbRoomHfGainEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->room_hf = farg(c, 0); }
HLE(_ZN2nn5audio24GetI3dl2ReverbRoomHfGainEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->room_hf : 0.0f); }
HLE(_ZN2nn5audio29SetI3dl2ReverbReflectionsGainEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->refl = farg(c, 0); }
HLE(_ZN2nn5audio29GetI3dl2ReverbReflectionsGainEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->refl : 0.0f); }
HLE(_ZN2nn5audio24SetI3dl2ReverbReverbGainEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->reverb = farg(c, 0); }
HLE(_ZN2nn5audio24GetI3dl2ReverbReverbGainEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->reverb : 0.0f); }
HLE(_ZN2nn5audio36SetI3dl2ReverbLateReverbHfDecayRatioEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->hf_ratio = farg(c, 0); }
HLE(_ZN2nn5audio36GetI3dl2ReverbLateReverbHfDecayRatioEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->hf_ratio : 0.0f); }
HLE(_ZN2nn5audio33SetI3dl2ReverbLateReverbDiffusionEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->diffusion = farg(c, 0); }
HLE(_ZN2nn5audio33GetI3dl2ReverbLateReverbDiffusionEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->diffusion : 0.0f); }
HLE(_ZN2nn5audio31SetI3dl2ReverbLateReverbDensityEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->density = farg(c, 0); }
HLE(_ZN2nn5audio31GetI3dl2ReverbLateReverbDensityEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->density : 0.0f); }
HLE(_ZN2nn5audio25SetI3dl2ReverbHfReferenceEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->hf_ref = farg(c, 0); }
HLE(_ZN2nn5audio25GetI3dl2ReverbHfReferenceEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->hf_ref : 0.0f); }
HLE(_ZN2nn5audio21SetI3dl2ReverbDryGainEPNS0_15I3dl2ReverbTypeEf) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->dry = farg(c, 0); }
HLE(_ZN2nn5audio21GetI3dl2ReverbDryGainEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); ret_float(c, rv ? rv->dry : 0.0f); }
HLE(_ZN2nn5audio33SetI3dl2ReverbLateReverbDecayTimeEPNS0_15I3dl2ReverbTypeENS_8TimeSpanE) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->decay_ns = (int64_t)ARG(1); }
HLE(_ZN2nn5audio33GetI3dl2ReverbLateReverbDecayTimeEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); RET(rv ? (uint64_t)rv->decay_ns : 0); }
HLE(_ZN2nn5audio33SetI3dl2ReverbReflectionDelayTimeEPNS0_15I3dl2ReverbTypeENS_8TimeSpanE) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->refl_delay_ns = (int64_t)ARG(1); }
HLE(_ZN2nn5audio33GetI3dl2ReverbReflectionDelayTimeEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); RET(rv ? (uint64_t)rv->refl_delay_ns : 0); }
HLE(_ZN2nn5audio33SetI3dl2ReverbLateReverbDelayTimeEPNS0_15I3dl2ReverbTypeENS_8TimeSpanE) { Reverb *rv = reverb_of(ARG(0)); if (rv) rv->late_delay_ns = (int64_t)ARG(1); }
HLE(_ZN2nn5audio33GetI3dl2ReverbLateReverbDelayTimeEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); RET(rv ? (uint64_t)rv->late_delay_ns : 0); }
HLE(_ZN2nn5audio21SetI3dl2ReverbEnabledEPNS0_15I3dl2ReverbTypeEb) {
    Reverb *rv = reverb_of(ARG(0));
    if (rv) rv->enabled = (int)(ARG(1) & 1);
    AUT("audio: i3dl2 reverb %s", ARG(1) & 1 ? "on" : "off");
}
HLE(_ZN2nn5audio20IsI3dl2ReverbEnabledEPKNS0_15I3dl2ReverbTypeE) { Reverb *rv = reverb_of(ARG(0)); RET(rv ? rv->enabled : 0); }
HLE(_ZN2nn5audio25SetI3dl2ReverbInputOutputEPNS0_15I3dl2ReverbTypeEPKaS4_i) {
    Reverb *rv = reverb_of(ARG(0));
    if (!rv) return;
    const int n = (int)ARG(3) > MAX_CH ? MAX_CH : (int)ARG(3);
    for (int k = 0; k < n; k++) {
        rv->in[k] = *(int8_t *)GP(ARG(1) + k);
        rv->out[k] = *(int8_t *)GP(ARG(2) + k);
    }
    rv->n = n;
    AUT("audio: i3dl2 reverb %d channels: in %d %d out %d %d", n, rv->in[0], rv->in[1], rv->out[0], rv->out[1]);
}
HLE(_ZN2nn5audio25GetI3dl2ReverbInputOutputEPKNS0_15I3dl2ReverbTypeEPaS4_Pii) {
    Reverb *rv = reverb_of(ARG(0));
    const int n = rv ? rv->n : 0;
    for (int k = 0; k < n && k < (int)ARG(4); k++) {
        *(int8_t *)GP(ARG(1) + k) = rv->in[k];
        *(int8_t *)GP(ARG(2) + k) = rv->out[k];
    }
    if (ARG(3)) G32(ARG(3)) = (uint32_t)n;
}
