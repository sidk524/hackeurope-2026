"use client";

import { useEffect, useRef } from "react";

type ThreeSceneProps = {
  className?: string;
};

export default function ThreeScene({ className = "" }: ThreeSceneProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Mount your Three.js renderer/scene/camera into `container`.
    // Ensure you clean up the renderer and any listeners on unmount.

    return () => {
      // Cleanup hook for Three.js resources.
    };
  }, []);

  return (
    <div
      className={`rounded-3xl border border-zinc-800 bg-zinc-950/60 p-4 shadow-lg ${className}`}
    >
      <div
        ref={containerRef}
        className="h-[420px] w-full rounded-2xl bg-zinc-900"
      />
    </div>
  );
}
