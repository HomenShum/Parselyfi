import React from "react";
import { Composition } from "remotion";
import { Demo, FeatureClip, FPS, WIDTH, HEIGHT, TOTAL_FRAMES } from "./Demo.jsx";
import { CLIPS } from "./clips.js";

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
  </>
);
