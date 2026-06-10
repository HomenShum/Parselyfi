import React from "react";
import { Composition } from "remotion";
import { Demo, FeatureClip, FPS, WIDTH, HEIGHT, TOTAL_FRAMES } from "./Demo.jsx";
import { CLIPS } from "./clips.js";
import { Walkthrough, WT_FPS, WT_W, WT_H, wtDuration } from "./Walkthrough.jsx";
import { WALKTHROUGHS } from "./walkthrough.data.js";

export const RemotionRoot = () => (
  <>
    <Composition
      id="Demo"
      component={Demo}
      durationInFrames={TOTAL_FRAMES}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
    />
    {CLIPS.map((c) => (
      <Composition
        key={c.id}
        id={c.id}
        component={FeatureClip}
        durationInFrames={c.durationInFrames}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
        defaultProps={{ seg: c }}
      />
    ))}
    {WALKTHROUGHS.map((w) => (
      <Composition
        key={"WT-" + w.id}
        id={"WT-" + w.id}
        component={Walkthrough}
        durationInFrames={Math.max(1, wtDuration(w))}
        fps={WT_FPS}
        width={WT_W}
        height={WT_H}
        defaultProps={{ wt: w }}
      />
    ))}
  </>
);
