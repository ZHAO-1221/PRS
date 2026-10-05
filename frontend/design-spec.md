# Pet Activity Report — Direction Study Specification

## Product and prototype scope

This is a desktop-first web prototype for one focused feature: a pet owner uploads a relatively long indoor recording, typically around thirty minutes, and receives a chronological report of what the dog was doing and how a person interacted with the dog. The prototype is not a complete pet-care product. It intentionally excludes account creation, pet profiles, camera pairing, notifications, subscriptions, social features, shopping, settings, and model administration. Its only story is upload, analysis, review, and playback of detected moments.

The temporary product name is **Pet Activity Report**. It is a functional working title rather than a proposed brand. All interface copy is English. The prototype uses a generated indoor feeding scene as explicitly labeled demo media because the user does not yet have a real video. The image must never be presented as model evidence or a real household recording.

## Audience and context

The primary audience is a dog owner reviewing a home-camera recording after being away. The secondary audience is an academic evaluator who needs to understand the relationship between the user experience and the underlying dog-behavior and human–dog interaction recognition system. The interface is viewed from approximately one metre away on a laptop or projected during a presentation. It therefore needs readable labels, visible time ranges, and an obvious cause-and-effect relationship between a selected event and video playback.

## Core user flow

1. Upload or select a thirty-minute indoor video.
2. Confirm the recording start time when metadata is unavailable. Clock time is essential because the report must say when an event happened, not merely show elapsed seconds.
3. Simulate analysis across detection and tracking, pose estimation, skeleton construction, dog-behavior recognition, and human–dog interaction recognition.
4. Show a report covering 09:00–09:30 with a video preview, two aligned timeline lanes, a chronological event list, confidence information, and a short natural-language interpretation.
5. Selecting an event seeks the video to that event’s start, visually marks its duration, updates the current label, and offers a play/pause control. The prototype simulates segment playback and stops at the selected event’s end.
6. A technical overlay toggle reveals or hides analysis annotations. The warmer owner-facing direction keeps it off by default; the research direction keeps it on by default.

## Shared content and data model

Both directions use identical sample content so the comparison is about design rather than data. The recording runs from 09:00 to 09:30. Dog-behavior events include Resting, Stand / Walk, Eat / Drink, Lie / Sleep, Play, and Resting. Human–dog interaction events include Feeding and Playing Together. The two tasks occupy separate timeline lanes because a dog behavior and an interaction can happen simultaneously. For example, Eat / Drink from 09:08–09:11 overlaps Feeding from 09:08–09:10. The report summary states that thirty minutes were analyzed and highlights two meaningful human interactions.

No diagnostic, veterinary, or safety conclusions are made. Confidence values are prototype sample data and are labeled as demo output. The interface should prefer statements such as “Your dog was eating while a person was feeding them” over technical model language, while still allowing the research direction to expose confidence, track IDs, and AI overlays.

## Direction A — Academic Research Console

Direction A is for a technical presentation or research demonstration. It uses cool off-white surfaces, deep navy, analytical blue, and a restrained orange highlight. The structure is a left navigation rail, a large video-analysis workspace, a right event inspector, and a dense two-lane timeline. Skeleton annotations are visible by default. Labels use compact uppercase metadata, precise confidence scores, track identifiers, and pipeline status. The layout should feel controlled, evidence-oriented, and credible, not like a generic dark cybersecurity dashboard.

## Direction B — Calm Pet Care Journal

Direction B is the recommended owner-facing direction. Its palette uses cream `#F6F1E8`, warm white `#FFFCF7`, cocoa `#60483D`, terracotta `#D97745`, sage `#7D8B72`, dark brown `#28231F`, and sand `#DED5C8`. The visual temperature is calm, warm, and trustworthy. The report reads like a carefully composed daily journal rather than an analytics console. Clock time and plain-language moments are primary; confidence and model annotations are secondary. The layout is more editorial, with a generous top summary, an immersive video area, and a chronological moment list. It must not become childish, cartoonish, or resemble a pet e-commerce site.

## Output and validation

Create two complete standalone HTML direction drafts at a shared 1440×900 comparison viewport. Each must remain usable at common desktop widths, include real clickable event selection, play/pause simulation, and overlay toggling, and produce zero console errors. Each direction receives one screenshot at 1440×900. The form is derived from the content: the thirty-minute recording becomes the horizontal temporal spine, while overlapping behavior and interaction lanes make the project’s two simultaneous recognition tasks visible.
