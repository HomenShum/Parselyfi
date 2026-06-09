import React from "react";
import { Composition } from "remotion";
import { Demo, FPS, WIDTH, HEIGHT, TOTAL_FRAMES } from "./Demo.jsx";

export const RemotionRoot = () => (
  <Composition
    id="Demo"
    component={Demo}
    durationInFrames={TOTAL_FRAMES}
    fps={FPS}
    width={WIDTH}
    height={HEIGHT}
  />
);
