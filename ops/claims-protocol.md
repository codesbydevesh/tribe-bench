# Claims Protocol — NeuroCheck Curation Methodology

This document governs how every claim in `neurocheck/claims_db/claims.yaml` is created.
The claims database is the intellectual core of the NeuroCheck paper. If the claims are
sloppy — wrong ROIs, unreplicable findings, bad citations — the paper is rejected and
the benchmark is worthless. Follow this protocol exactly.

---

## What Makes a Valid Claim

A claim is a neuroscience finding that:
1. Has been replicated at least once beyond the original paper
2. Can be expressed as a contrast: "stimulus A activates region X more than stimulus B"
3. Maps to a specific region in the Glasser HCP-MMP1 atlas
4. Can be tested with stimuli available in public datasets
5. Has a known expected effect size (or we can estimate one from the literature)

## What Is NOT a Valid Claim

- Findings from a single unreplicated study
- Claims about connectivity or temporal dynamics (TRIBE v2 predicts activation, not connectivity)
- Claims about subcortical structures (TRIBE v2 outputs cortical surface only)
- Claims requiring subject-specific anatomy (TRIBE v2 predicts average subject)
- Claims about attention, memory, or cognitive state (TRIBE v2 takes stimuli, not cognitive states)
- Claims from clinical populations (TRIBE v2 trained on healthy adults only)

---

## The 9-Step Curation Process

### Step 1: Identify Candidate Claim
Source: neuroscience textbooks, review papers, meta-analyses.
Prefer claims from meta-analyses (strongest evidence).
Record: plain-English claim, original citation, year.

### Step 2: Verify Replication
Find at least one independent replication. Check:
- Different research group than original authors
- Published in peer-reviewed journal
- Effect replicated in same direction with significance

If no replication exists, reject the claim.

### Step 3: Map to Glasser Atlas Region
Identify the brain region in the original papers (often in Talairach/MNI coordinates
or Brodmann areas). Map to the corresponding Glasser HCP-MMP1 region.

Mapping resources:
- Glasser et al., 2016 (Nature) — the atlas paper with region descriptions
- nilearn documentation for region labels
- NeuroSynth (neurosynth.org) — coordinate-to-region lookup

Record: Glasser region label(s), hemisphere(s).

If the finding involves a region not clearly mappable to Glasser, reject or mark
as "approximate" with explanation.

### Step 4: Define Contrast Direction
Express as: "stimulus A should produce higher activation in region X than stimulus B."
The direction must be unambiguous from the literature.

### Step 5: Identify Public Stimulus Sources
Both stimulus A and stimulus B must come from publicly available datasets.
No stimuli behind paywalls, institutional access, or custom-created stimuli.

Common sources:
- CelebA / CelebA-HQ — face images
- Places365 — scene/building images
- COCO — general objects, people, animals
- ImageNet — object categories
- AudioSet — audio clips by category
- LibriSpeech — speech audio
- Kinetics-700 — video clips by action category
- UCF-101 — action recognition video clips

Record: dataset name, category/class, any filtering criteria.

For video stimuli: TRIBE v2 expects video input. Static images must be converted
to short videos using tribe_tools.video_utils.image_to_video().

### Step 6: Estimate Expected Effect Size
From the literature (ideally meta-analysis), record:
- Cohen's d (if available)
- If not available, estimate from reported t-statistics and sample sizes
- Set a minimum threshold (default: d=0.3, medium effect)

Claims with very small expected effects (d < 0.2) are harder to detect and
should be marked difficulty="hard".

### Step 7: Assign Difficulty
- **easy**: Large effect (d > 0.5), well-replicated (5+ studies), clear ROI mapping
- **medium**: Moderate effect (0.3-0.5), replicated (2-4 studies), reasonable ROI mapping
- **hard**: Small effect (0.2-0.3), fewer replications, approximate ROI mapping

### Step 8: Write YAML Entry
Follow the schema in interface-contracts.md exactly. Include all fields.

### Step 9: Cross-Check
Before finalizing, verify:
- [ ] Citation DOI resolves to the correct paper
- [ ] Replication citation is from a different group
- [ ] Glasser region label exists in the atlas (check against region list)
- [ ] Stimulus datasets are still publicly accessible
- [ ] Effect direction matches what the paper actually reports (not our interpretation)
- [ ] No overlap/redundancy with existing claims in the database

---

## Categories

Organize claims into these categories (aim for balanced coverage):

| Category | Target Count | Examples |
|----------|-------------|----------|
| visual_selectivity | 10 | Faces>houses in FFA, scenes>objects in PPA |
| auditory_processing | 8 | Speech>noise in STG, music>speech in right auditory |
| language | 8 | Sentences>wordlists in IFG, semantic>phonological |
| multimodal | 6 | AV speech>A-only in STS, reading>listening in visual |
| motor_perception | 5 | Biological motion>random in pSTS, hands>faces in EBA |
| emotion | 5 | Fearful>neutral faces in amygdala-adjacent cortex |
| attention_networks | 4 | Salient>nonsalient in TPJ |
| high_level_cognition | 4 | Theory of mind>physical in TPJ/mPFC |

Total: 50 claims

---

## Quality Checklist (Run for Every Batch of Claims)

After adding a batch of claims, run this checklist:

1. Load YAML and verify it parses without errors
2. Check all region names exist in Glasser atlas (script: cross-reference with atlas.py)
3. Check all DOIs resolve (script: HTTP HEAD request to doi.org)
4. Check for duplicate claims (same region + same contrast direction)
5. Check category balance (no category should exceed target by more than 2)
6. Verify no claim requires subcortical regions
7. Verify no claim requires subject-specific anatomy
8. Verify all stimulus sources are public datasets (not institutional)

---

## When In Doubt

- Prefer well-established findings over novel ones
- Prefer meta-analyses over single studies
- Prefer larger effect sizes over smaller ones
- Prefer visual/auditory claims (TRIBE v2's strengths) over high-level cognition
- When the Glasser mapping is ambiguous, use the broader network assignment
  (e.g., "visual network" rather than a specific subregion)
- Ask the human before including any claim marked difficulty="hard"
