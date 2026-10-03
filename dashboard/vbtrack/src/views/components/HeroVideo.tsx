import { useState } from 'react'

interface Props {
  src?: string
  width: number
  height: number
  label: string
}

// Sized to the real capture resolution (1280x720 match footage) via the
// aspect-ratio box below, rather than an arbitrary hero banner ratio.
// With no src, or if the file fails to load, it falls back to a blank
// Swiss-chrome placeholder at the same size instead of breaking layout.
export function HeroVideo({ src, width, height, label }: Props) {
  const [errored, setErrored] = useState(false)
  const showVideo = Boolean(src) && !errored

  return (
    <section className="hero-video" style={{ aspectRatio: `${width} / ${height}` }}>
      {showVideo ? (
        <video
          key={src}
          className="hero-video-el"
          src={src}
          autoPlay
          loop
          muted
          playsInline
          controls
          onError={() => setErrored(true)}
        />
      ) : (
        <div className="hero-video-blank">no footage loaded</div>
      )}
      <span className="hero-video-tag">{label}</span>
    </section>
  )
}
