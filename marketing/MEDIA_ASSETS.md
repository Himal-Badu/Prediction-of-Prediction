# MEDIA_ASSETS.md — PoP Content Ideas & Assets

> **Purpose:** Ready-to-use content ideas, visual concepts, and asset specs for X posts.  
> **Tone:** Technical but accessible. Show, don't just tell. Visual > text-only.

---

## 1. PoP Architecture — ASCII Diagrams

### Core Architecture (For X posts)

```
┌─────────────────────────────────────────────┐
│                  PoP Layer                  │
│  ┌────────────────────────────────────────┐ │
│  │     Meta-Attention Head Network        │ │
│  │  ┌──────────┐  ┌──────────┐           │ │
│  │  │ Attention │  │ Gradient │           │ │
│  │  │ Pattern   │→ │ Flow     │→ Confidence│ │
│  │  │ Analyzer  │  │ Monitor  │   Score   │ │
│  │  └──────────┘  └──────────┘           │ │
│  └────────────────────────────────────────┘ │
│                    ↕                         │
│  ┌────────────────────────────────────────┐ │
│  │        Detection Output                │ │
│  │  ✅ Pass → Generate normally           │ │
│  │  ⚠️  Flag → Caution mode              │ │
│  │  ❌ Block → Refuse & explain           │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────┐
│              Base LLM (Any)                 │
│  GPT │ Claude │ Llama │ Gemini │ Mistral   │
└─────────────────────────────────────────────┘
```

### How It Works (Simplified)

```
Input Prompt
    ↓
┌─────────┐    Single Pass    ┌─────────────┐
│ Base LLM │────────────────→ │ PoP Layer   │
│ Forward  │  (attention map) │ Analysis    │
└─────────┘                  └──────┬──────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                     ✅ Generate         ⚠️ Flag
                     with confidence     & correct
```

### The 83% Precision Claim

```
┌──────────────────────────────────┐
│  Hallucination Detection Methods │
│                                  │
│  Traditional Eval:    ████░░░ 45%│
│  RAG + Eval:          ██████░ 68%│
│  Multi-pass Check:    ███████░ 72%│
│  PoP (single-pass):   ████████ 83%│ ← us
│                                  │
│  █ = Precision    ░ = Error      │
└──────────────────────────────────┘
```

### Meta-Learning Explained Simply

```
Standard LLM:
  Input → [LLM] → Output → "Is this right?" → Hope

PoP:
  Input → [PoP sees attention patterns]
       → "This is likely wrong BEFORE you generate"
       → [LLM] → Corrected Output
```

---

## 2. Screenshot Ideas

### Benchmark Results Screenshots

| Shot | Content | When to Post |
|------|---------|-------------|
| **Training curve** | Loss decreasing, accuracy rising over epochs. Annotate key milestones. | Week 1–2, build-in-public |
| **Confusion matrix** | TP/FP/TN/FN for hallucination detection. Show PoP vs baseline. | Week 3, technical thread |
| **Comparison table** | PoP vs other methods. Columns: Precision, Recall, Latency, Single-pass? | Month 1, viral thread |
| **Real-time demo** | Terminal/browser showing PoP catching a hallucination in real-time | Month 2, demo video |
| **Latency chart** | PoP adds <Xms overhead vs multi-pass methods at 10x the time | Month 2, technical thread |
| **Error heatmap** | Visualization of which tokens PoP flagged as high-risk | Month 3, fundraising content |

### Visual Design Tips for Screenshots

- **Dark mode terminal** — looks 10x better than light mode
- **Add annotations** — arrows, circles, "👆 this is what matters"
- **Crop tight** — show only the relevant part, no desktop clutter
- **Include your cursor** — feels real, not staged
- **Terminal font** — use JetBrains Mono or Fira Code

---

## 3. Short Video Ideas

### Demo Walkthroughs (60–90 seconds)

| Video | Script | Hook |
|-------|--------|------|
| **"PoP catches a hallucination"** | Show a prompt → LLM starts generating wrong info → PoP flags it → Correct answer shown | "Watch PoP catch a hallucination in 3 seconds" |
| **"Why LLMs are wrong"** | Visualize attention patterns → Show where the model "decides" to hallucinate → PoP detects this pattern | "LLMs hallucinate because of THIS pattern in their attention" |
| **"PoP vs ChatGPT"** | Same prompt → ChatGPT hallucinates → PoP catches it → Side by side | "I gave ChatGPT a trick question. PoP caught what it got wrong." |
| **"Building PoP in 60 seconds"** | Time-lapse of code → training → results | "Building an AI that catches AI mistakes, in 60 seconds" |
| **"The $24.6B problem"** | Market visualization → Why hallucinations matter → PoP as solution | "This one problem costs $24.6B. Here's how we're fixing it." |

### Video Production Tips

- **Screen recording** with voiceover (Loom, QuickTime, OBS)
- **Subtitles always** — 85% of X videos are watched on mute
- **First 3 seconds matter most** — lead with the result, not the setup
- **Keep it under 90 seconds** — X algo penalizes long videos
- **End with CTA** — "Follow for more" or "Link in bio"

### Equipment Needed (Low Budget)
- Screen recorder: OBS (free) or Loom (free tier)
- Voiceover: Phone mic or laptop mic (good enough for X)
- Editing: CapCut (free) or DaVinci Resolve (free)
- Thumbnails: Canva (free)

---

## 4. Meme & Relatable Content Ideas

### Hallucination Memes

**Meme 1: "LLM confidently wrong"**
> Image: Dog sitting in burning room saying "This is fine"
> Text: "Me asking GPT about a topic I know well"
> Subtext: "PoP: 🚨 detected 4 errors before generation"

**Meme 2: "Architect drawing"**
> Image: Guy drawing on whiteboard
> Labeled: "ML Engineers" "adding another fine-tuning layer" "to fix hallucinations"
> Subtext: Meanwhile PoP just reads the attention map 🤷

**Meme 3: "Trust"**
> Format: Drake meme
> Top: "Trusting the LLM's output"
> Bottom: "Trusting a layer that checks the LLM's output"
> Caption: "PoP: because confidence isn't correctness"

**Meme 4: "You guys are getting paid"**
> Image: "Wait, you guys are getting paid?"
> Text: "Wait, your model actually knows when it's wrong?"
> Caption: "PoP: making LLMs self-aware since 2026"

**Meme 5: "Before PoP vs After PoP"**
> Before: Chaos, fire, hallucinations everywhere
> After: Calm, accurate, reliable
> Text: "The difference one layer makes"

### Build-in-Public Memes

**"My git log this week"**
```
Monday:     "feat: added meta-attention heads"
Tuesday:    "fix: training crashed after epoch 47"
Wednesday:  "feat: 83% precision on hallucination dataset"
Thursday:   "fix: forgot to save model checkpoint 💀"
Friday:     "feat: it works (for now)"
```

### Contrarian/AI Culture Memes

**"Hot take"**
> "Unpopular opinion: RAG is duct tape. PoP is surgery."
> (Gets engagement from both RAG fans and haters)

**"AI founder life"**
> "Everyone: 'What do you do?'
> Me: 'I built a neural network that catches when other neural networks lie'
> Everyone: '...so you're a lie detector for AI?'
> Me: '...sure, let's go with that'"

---

## 5. Visual Brand Guide

### Color Palette
- **Primary:** #00D4FF (electric blue — tech, trust, innovation)
- **Secondary:** #7C3AED (purple — AI, neural networks)
- **Accent:** #10B981 (green — correct, pass, good)
- **Warning:** #F59E0B (amber — flag, caution)
- **Error:** #EF4444 (red — hallucination detected)
- **Background:** #0F172A (dark navy — dark mode standard)

### Typography for Graphics
- **Headlines:** Inter Bold or SF Pro Bold
- **Code:** JetBrains Mono or Fira Code
- **Body:** Inter Regular

### Post Image Specs
- **Single image:** 1200 x 675 px (16:9)
- **Thread header:** 1200 x 675 px
- **Profile picture:** 400 x 400 px
- **Banner:** 1500 x 500 px
- **Format:** PNG for diagrams, JPEG for photos

---

## 6. Content Calendar Template

### Weekly Content Mix

| Day | Content Type | Format | Topic |
|-----|-------------|--------|-------|
| **Mon** | Technical thread | Text + ASCII | Deep-dive on PoP internals |
| **Tue** | Engagement | Replies | 10 thoughtful replies to target accounts |
| **Wed** | Build-in-public | Screenshot + text | Honest progress update |
| **Thu** | Data/visual | Image/video | Benchmark, architecture, or training viz |
| **Fri** | Hot take / meme | Text or image | Contrarian opinion or relatable AI content |
| **Sat** | Light engagement | Replies | Reply to DMs, engage with community |
| **Sun** | Planning | — | Draft next week's content |

### Content Bank (Pre-Write These)

1. ✏️ "How PoP works" thread (detailed)
2. ✏️ "Why LLMs hallucinate" thread
3. ✏️ "PoP vs traditional eval" comparison
4. ✏️ "Building PoP — Week 1" build-in-public
5. ✏️ "The $24.6B hallucination problem" market thread
6. ✏️ "I'm 16 and I built..." origin story thread
7. ✏️ 3–5 standalone hot takes
8. ✏️ 2–3 meme ideas (have images ready)

---

## 7. Metrics to Track (Content Performance)

| Content Type | Target Engagement | Track |
|-------------|-------------------|-------|
| Technical thread | 500+ likes, 50+ RT | Which topics resonate |
| Build-in-public | 200+ likes | Do people care about progress? |
| Hot takes | 300+ likes, 100+ replies | Which angles generate debate? |
| Memes | 500+ likes | Humor reaches new audiences |
| Demos/Videos | 1000+ views | Video is underused = opportunity |
| Replies | 5+ likes per reply | Quality of engagement |

### What Works Best (X Algorithm in 2026)
1. **Threads** > single tweets (more dwell time)
2. **Images** > text-only (more engagement)
3. **Video** > images (highest reach, but harder to make)
4. **Polls** drive replies (algorithm loves engagement)
5. **Questions** at end of posts increase replies
6. **First 30 minutes** determine reach (reply to early comments fast)
