# Subtitle Generator

A Python application that converts speech in an **audio or video file** into **timestamped subtitles (SRT and WebVTT)** using a ready-made speech-to-text model, **faster-whisper** (an optimized implementation of OpenAI's Whisper). Everything runs **locally**: no audio is uploaded to any online service.

Built for the *Applied AI Programming (TX00FM14)* course at Metropolia University of Applied Sciences.

**Demo video:** _add link here_

Example output (`examples/clear_speech_prompt.srt`):

```
1
00:00:01,460 --> 00:00:06,860
Hello, my name is Bibas Dhital and I study
Smart Automation at Metropolia University

2
00:00:06,860 --> 00:00:09,160
of Applied Sciences in Helsinki.
```

---

## Features

- **Input:** audio (WAV, MP3, M4A, FLAC, OGG, AAC) and video (MP4, MKV, MOV, WEBM, AVI), a single file or a whole folder
- **Output:** **SRT** and/or **WebVTT** subtitles, plus a plain-text transcript and a CSV quality report for every file
- **Word-level timestamps** grouped into **readable cues**: max 2 lines × 42 characters, 1–6 seconds per cue, max 20 characters/second reading speed, new cue after a pause
- **Long input** is split into many cues (a 9.6-minute video produced 86 cues)
- **Automatic language detection**, or a chosen language (`--language`)
- **Translation** of non-English speech into English subtitles (`--translate`)
- **Vocabulary hints** for names and technical terms (`--prompt`)
- **Selectable model size** (`tiny` … `large-v3`) and CPU or NVIDIA GPU
- **Silence filter (VAD)** that can be tuned or switched off
- **Quality information:** detected language and its probability, average word confidence, low-confidence cues, longest line, shortest cue, fastest reading speed
- **Clear error messages** for missing files, unsupported formats, unknown or missing model, and failed transcription
- **Offline mode** (`--offline`) that guarantees nothing is downloaded

---

## Setup

Tested on Windows 11 with Python 3.12.10, faster-whisper 1.2.1 and PyAV 18.1.0, on a laptop CPU.

```bash
# 1. Get the code
git clone https://github.com/Bibas41/subtitle-generator.git
cd subtitle-generator

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 3. Install the dependencies
pip install -r requirements.txt
```

The **first run downloads the chosen Whisper model** (about 480 MB for `small`) from Hugging Face. After that it works fully offline. No separate FFmpeg installation is needed; faster-whisper decodes audio and video with the PyAV library.

> **Important version note:** `pip install faster-whisper` currently installs **PyAV 19**, which is **not compatible** with faster-whisper 1.2.1: every audio file fails to load (`open() got an unexpected keyword argument 'metadata_errors'`). I found this during testing. `requirements.txt` therefore pins **`av==18.1.0`**, which I verified with MP3, WAV, M4A and MP4 files.

---

## Usage

```bash
# One file, default settings (small model, SRT + WebVTT, automatic language)
python subtitle_generator.py --input examples/tts_clear_speech.mp3

# Whole folder
python subtitle_generator.py --input test_media

# Show each recognized segment while transcribing
python subtitle_generator.py --input lecture.mp4 --verbose

# Help Whisper with names and technical terms
python subtitle_generator.py --input talk.m4a --prompt "Bibas Dhital, Metropolia, SRT, WebVTT"

# Nepali speech -> English subtitles
python subtitle_generator.py --input nepali.m4a --translate

# Conversation with a quiet or distant speaker: switch off the silence filter
python subtitle_generator.py --input interview.m4a --no-vad
```

Results are saved in `output/` as `<name>.srt`, `<name>.vtt`, `<name>.txt` (transcript) and `<name>_report.csv` (every cue with timing, characters per second and confidence).

### Options

| Option | Default | Meaning |
|---|---|---|
| `--input` | – | Audio/video file or folder |
| `--model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, or a path to a local model |
| `--language` | auto | Language code, e.g. `en`, `fi`, `ne` |
| `--translate` | off | Translate the speech into English subtitles |
| `--prompt` | – | Vocabulary hints (names, technical terms) |
| `--format` | `both` | `srt`, `vtt` or `both` |
| `--no-vad` | off | Do not remove silent parts before transcription |
| `--vad-threshold` | `0.5` | Silence filter sensitivity (lower keeps quieter sounds) |
| `--no-word-timestamps` | off | Use segment timestamps only (faster, rougher splitting) |
| `--device` | `cpu` | `cpu` or `cuda` (NVIDIA GPU) |
| `--offline` | off | Never download; use an already downloaded model |
| `--suffix` | – | Add text to output names, e.g. `_prompt`, to compare settings |
| `--output` | `output/` | Output folder |
| `--verbose` | off | Print each segment while transcribing |

---

## How it works

```
audio / video file
      │
      ▼
decode audio (PyAV, inside faster-whisper) ── 16 kHz mono
      │
      ▼
silence filter (Silero VAD) ── removes non-speech parts
      │
      ▼
Whisper "small" model (CTranslate2, int8 on CPU)
      │   → text segments + a start/end time and probability for every word
      ▼
group words into cues (readability rules) ──► fix timing (min 1 s, max 20 chars/s, no overlap)
      │
      ▼
wrap into ≤ 2 lines ──► write SRT, WebVTT, transcript, quality report
```

### Model choice

I used **faster-whisper** with the **Whisper `small`** model:

- It is in the course material, **free and open source** (MIT licence), and needs no training.
- It runs **locally on a laptop CPU**: about **2.5–4.5× faster than real time** in my tests with `int8` quantization. No GPU and no cloud service are needed, which also matters for privacy.
- `small` (≈480 MB) is a compromise: `tiny` and `base` are faster but noticeably less accurate, while `medium` and `large-v3` are more accurate but much slower on a CPU. The model can be changed with `--model`.
- It is multilingual, detects the language automatically, and can translate into English.

### How timestamps are produced and formatted

1. Whisper returns **segments** (roughly sentence-sized pieces). With `word_timestamps=True` it also estimates a **start and end time for every word**, plus a probability.
2. The program groups words into **cues** and starts a new cue when:
   - the next word would make the cue longer than **84 characters** (2 lines × 42),
   - the cue would stay on screen longer than **6 seconds**,
   - there is a **pause of more than 0.8 s** before the word,
   - or a **sentence ends** (`.`, `?`, `!`), but not after single-letter abbreviations like "B.".
3. **Very short cues** (1–2 words shown for under 1 second, e.g. a lone "files.") are **merged** with a neighbouring cue if the result still fits the limits and stays within one sentence.
4. **Timing is adjusted for readability:** each cue stays on screen at least **1 second** and long enough to read at **20 characters per second**, but never overlaps the next cue.
5. Each cue is **wrapped into at most two balanced lines** of ≤ 42 characters, preferring a break after a comma.
6. Times are written as `HH:MM:SS,mmm` in **SRT** (numbered cues, **comma** before milliseconds) and as `HH:MM:SS.mmm` in **WebVTT** (file starts with `WEBVTT`, **period** before milliseconds).

The readability limits follow common subtitle guidelines (one or two lines, no very long lines, no very short display times); they are constants at the top of `subtitle_generator.py` and easy to change.

---

## Testing

All recordings were made by me on a laptop microphone for this project (Windows Sound Recorder), except the TTS file, which was generated with the text-to-speech (listen) feature of **Google Translate** from the same script text. The person in the conversation test agreed to be recorded. For accuracy tests I read a **fixed script** and compared the subtitles with it word by word. Word error rates below are **approximate manual counts** (substitutions + deletions + insertions, about 53 words); numbers written as digits ("2026") were counted as correct.

**Script:** *"Hello, my name is Bibas Dhital and I study Smart Automation at Metropolia University of Applied Sciences in Helsinki. This recording is a test for my subtitle generator. It uses the Whisper speech recognition model, and it saves the subtitles as SRT and WebVTT files. Today is Wednesday, the thirtieth of September, twenty twenty-six."*

### Test set

| File | Content | Tests |
|---|---|---|
| `clear_speech.m4a` | Script, my voice, quiet room | baseline + non-native accent |
| `tts_clear_speech.mp3` | Same script, synthetic TTS voice | accent vs. vocabulary |
| `noisy.m4a` | Same script, music playing loudly nearby | background noise |
| `pauses.m4a` | Sentences with 3–5 s pauses, a lone "Yes.", one fast sentence | pauses, short cues, fast speech |
| `two_speakers.m4a` | 40 s conversation with a friend, one interruption | multiple speakers, overlap |
| `nepali.m4a` | About 25 s of Nepali | language detection, non-English, translation |
| `face_anonymizer_demo.mp4` | My 9.6-minute demo video from another project | long input, video format |

### Results overview

| File | Language detected | Cues | Speed (× real time) | Avg. word confidence | Approx. WER |
|---|---|---|---|---|---|
| clear_speech | en (0.86) | 6 | 2.4–2.8× | 0.85 | **≈ 19 %** |
| clear_speech + `--prompt` | en (0.86) | 6 | 2.6× | 0.82 | **≈ 6 %** |
| tts_clear_speech | en (0.99) | 6 | 2.8× | 0.91 | ≈ 8 % |
| noisy | en (0.77) | 6 | 3.1× | 0.82 | ≈ 25 % |
| pauses | en (0.93) | 7 | 4.3× | 0.77 | – |
| two_speakers | en (0.97) | 7 | 4.0× | 0.91 | one whole line missing |
| nepali | **ne (0.95)** | 5 | 0.9× | 0.73 | poor (see below) |
| nepali `--translate` | ne (0.95) | 7 | 3.1× | 0.61 | meaning partly wrong |
| face_anonymizer_demo.mp4 (9.6 min) | en (0.97) | **86** | 4.5× | 0.82 | good; errors in names/terms |

### 1. Transcription errors: accent or vocabulary?

The same script was spoken by me and by a TTS voice:

| Word(s) | My voice | + `--prompt` | TTS voice | Cause |
|---|---|---|---|---|
| Bibas Dhital | B. U. S. Kital | **Bibas Dhital** | Bebas Duttal | **rare name**: fails for both voices, fixed by the prompt |
| Metropolia | Metropolitan | **Metropolia** | Metropolitan | **rare word**: fails for both voices, fixed by the prompt |
| speech recognition | SP's Recognition | Space Recognition | ✅ | **pronunciation/accent** |
| subtitles | subtitle | subtitle | ✅ | pronunciation/accent |
| SRT / WebVTT | SIT / WebBTT | **SRT / WebVTT** | SRT / web VTT | abbreviations: accent + vocabulary |
| thirtieth | **13th** | **13th** | 30th | **pronunciation/accent**: changes the meaning |

**Findings:**
- **Names and rare words** fail regardless of the speaker. A short **vocabulary prompt** fixed all of them (≈ 19 % → ≈ 6 % WER).
- **Technical terms, abbreviations and numbers** failed mainly with my **non-native accent**; the TTS voice got them right.
- The prompt could not fix words the model *hears* differently: "speech" became "Space" even though "speech recognition" was in the prompt.
- **"thirtieth" → "13th"** is the most important error: a small mistake that **changes the meaning** (a wrong date). Errors like this are why automatic subtitles should be reviewed before anyone relies on them.

### 2. Background noise

With music playing, the WER rose from ≈ 19 % to ≈ 25 %, and the errors became **plausible-sounding wrong words**: "subtitle generator" → **"software editor"**, "subtitles" → "software itself", "speech recognition" → "Whispers-P's recommendation". Language-detection confidence dropped from 0.86 to 0.77. Surprisingly, the date was correct this time ("30th"), which shows how **unpredictable** the errors are. Timing was not affected.

### 3. Pauses, short cues and fast speech

- **No subtitles are shown during silences**, and every sentence became its own cue.
- The lone **"Yes"** stays on screen for **1.0 s** (the minimum-duration rule), otherwise it would flash by.
- The **fast sentence** was split into two readable cues but had small errors ("speak" → "say", "should still" → "should is still").
- At the end I said "Thank you" **twice** (with a 2-second pause between); both were correctly recognized as **two separate cues**.

### 4. Two speakers and the silence filter

| Setting | Friend's line "No problem. What do I have to do?" | Interruption | Pause timing | Cues |
|---|---|---|---|---|
| Silence filter ON (default, 0.5) | **missing** | both voices merged into one sentence | ✅ correct | 7 |
| More sensitive filter (0.25) | still missing | merged | ✅ correct | 8 |
| Silence filter OFF (`--no-vad`) | ✅ **recovered** | ✅ split into two cues | – | 11 |

A loudness analysis of the recording showed why: my friend was **10–15 dB quieter** (about −48 to −55 dB vs. −40 dB), and the silence filter treated the quiet voice as background and **removed it before Whisper heard it**.

But switching the filter off has a cost. On `pauses.m4a` with `--no-vad`:
- subtitles **stayed on screen through the silences**,
- sentences were **glued together across a 4-second pause** ("This is a test with pauses did the"),
- "Yes" was shown for **4.4 s**, mostly *before* it was spoken,
- all punctuation disappeared, and confidence fell from 0.77 to 0.64.

**Decision:** the silence filter stays **ON by default** (better timing for normal recordings). For conversations with **quiet or distant speakers**, `--no-vad` is recommended. A more sensitive threshold (0.25) was tested and did not help.

Also: **Whisper does not identify speakers.** The subtitles never show who is talking, and overlapping speech is merged. For viewers who rely on subtitles, this makes conversations hard to follow.

### 5. Nepali and translation

- The language was **correctly detected as Nepali** (probability 0.95).
- The **Nepali transcription was poor**: Devanagari script, but many misspelled or Hindi-like words ("नमस्टे", "भीवर्स दिताल") and a stray one-letter cue ("प्"). It was also the slowest file (0.9× real time).
- The **English translation was surprisingly understandable** ("My home is Nepal", "Gorkha district", "Helsinki") but had **meaning errors**: *बुझाउनु* ("to submit") was translated as "to **understand**", giving "I have to *understand* this project in the middle of the night", and "study at Metropolia" became "doing smart automation in the metropolis".
- Whisper was trained on much less Nepali than English, so quality depends strongly on the language. This is a **fairness and accessibility issue**: speakers of less-resourced languages get worse subtitles.

### 6. Long video and synchronization

The 9.6-minute demo video produced **86 cues** in 129 s (4.5× real time) with average confidence 0.82. I loaded the SRT file in **Windows Media Player** (which confirms the file format is valid) and watched parts of it with the subtitles on:

- The subtitles **appeared and disappeared in sync with the speech** and were comfortable to read.
- Most sentences were word-for-word correct.
- Errors again appeared in **names and technical terms**: "MediaPipe" → "media file", "VS Code" → "VES code", "GitHub repo" → "github rip", "BlazeFace" → "blaze face", and my self-introduction became "Lomati".

### 7. Error handling

| Test | Command | Output |
|---|---|---|
| Missing file | `--input test_media\missing.mp3` | `[ERROR] Input not found: test_media\missing.mp3` |
| Unsupported format | `--input error_tests\notes.txt` | `[ERROR] Unsupported file type: .txt` + list of supported types |
| Unknown model | `--model huge` | `[ERROR] Unknown model 'huge'. Use one of tiny, base, small, medium, large-v3 ...` |
| Missing model | `--model medium --offline` | `[ERROR] Could not load model 'medium' ...` + `Hint: the model is not downloaded yet - run once without --offline` |
| Failed transcription | `--input error_tests\broken.mp3` (a text file renamed to .mp3) | `[ERROR] Transcription failed for broken.mp3: Invalid data found when processing input` + possible causes |
| No speech | (tested in code) | `[WARN] No speech detected - no subtitle file written.` |

Input problems are detected **before** the model is loaded, so they fail instantly. In every case the program printed a clear message and did not crash.

### 8. Problems found and fixed during testing

| Problem found | Fix |
|---|---|
| PyAV 19 broke all audio loading | pinned `av==18.1.0` in `requirements.txt` |
| One-word cues flashing for 0.48 s ("files.") or 0.74 s ("University") | merge very short cues with a neighbour; 8 → 6 cues on `clear_speech` |
| "B." in "B. U. S." treated as a sentence end | single-letter abbreviations no longer end a cue |
| Lines of 45–48 characters (limit 42) | line breaks are only chosen where both lines fit |
| A translated cue of 69 characters shown for 1.8 s (≈ 38 characters/s) | reading-speed rule: max 20 characters/s → now 3.5 s |

---

## What kinds of audio work well

- **One speaker close to the microphone** in a quiet room
- **Common vocabulary** in English (and other widely spoken languages)
- **Normal to moderately fast** speech, with or without pauses
- **Long recordings and videos**: tested with 9.6 minutes, 86 cues
- Names and technical terms **when given as a `--prompt`**

## What kinds of audio are difficult

- **Names, rare words, abbreviations and technical terms** (without a prompt)
- **Non-native accents** for technical terms and numbers (e.g. "thirtieth" → "13th")
- **Background noise or music**: plausible but wrong words
- **Quiet or distant speakers**: can be removed completely by the silence filter
- **Several people**: no speaker labels, overlapping speech is merged
- **Less-resourced languages** such as Nepali: poor transcription, partly wrong translation

---

## Limitations

- Transcripts are **not guaranteed correct**; errors can change the meaning (dates, numbers, names).
- **No speaker identification** (diarization).
- Timestamps come from the model's estimates; they are accurate to a fraction of a second, which is fine for subtitles but not for precise editing.
- The silence filter is a **trade-off** between clean timing and not losing quiet speech.
- Only tested on one laptop, one microphone, and my own voice, one other person and one TTS voice. Accuracy numbers are small manual counts, not a benchmark.
- No manual editing step: corrections have to be made in the `.srt` file with a text editor.
- **Possible improvements:** loudness normalization before transcription (to rescue quiet speakers), a larger model on a GPU, speaker diarization, burning subtitles into the video with FFmpeg, and a simple correction interface.

---

## Privacy, consent and accessibility

- **Local processing:** the model runs on the laptop. Audio and transcripts are never sent to a cloud service. After the one-time model download, `--offline` guarantees no network access.
- **Consent:** all recordings are my own; the second speaker agreed to take part. For the TTS file, only the (non-sensitive) script text was entered into Google Translate; no recordings were uploaded.
- **No original recordings are published.** Voices can identify people, so the `test_media/` folder is excluded from this repository with `.gitignore`. Only the **synthetic TTS audio** is included as a demo input, plus the generated subtitle files (text only).
- **Transcripts are personal data too:** subtitles can reveal names, places or opinions. Output files should be stored only as long as needed and shared with care.
- **Accessibility:** subtitles help deaf and hard-of-hearing viewers, people watching without sound, and language learners. But **inaccurate subtitles can mislead exactly the people who rely on them** (e.g. a wrong date, a missing line, or no speaker names). Automatic subtitles should be **reviewed and corrected by a human** before being published for accessibility. The quality report (`_report.csv`) helps with this by flagging low-confidence cues.

---

## Project structure

```
subtitle-generator/
├── subtitle_generator.py   # the application
├── requirements.txt        # faster-whisper + pinned PyAV version
├── examples/               # demo input (synthetic TTS audio) and generated subtitles
├── README.md
└── .gitignore              # keeps .venv/, test_media/, output/ and error_tests/ out of Git
```

---

## Sources

- OpenAI Whisper repository: https://github.com/openai/whisper
- Whisper model card: https://github.com/openai/whisper/blob/main/model-card.md
- faster-whisper repository: https://github.com/SYSTRAN/faster-whisper
- FFmpeg documentation: https://www.ffmpeg.org/documentation.html
- Python `wave` module: https://docs.python.org/3/library/wave.html
- Library of Congress, SubRip (SRT) format: https://www.loc.gov/preservation/digital/formats/fdd/fdd000569.shtml
- W3C WebVTT standard history: https://www.w3.org/standards/history/webvtt1/
- MDN, WebVTT API: https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API
- MDN, WebVTT format: https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API/Web_Video_Text_Tracks_Format
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper paper, 2022): https://arxiv.org/abs/2212.04356

**AI assistance:** As encouraged in the course, I used an AI assistant (Claude) to help write and debug the code and structure this README. The recordings, testing, observations and decisions were made and checked by me.
