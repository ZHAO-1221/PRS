// Uploaded clips have no cover image yet, so their first video frame stands in.
export default function Cover({ clip, alt, className = 'cover' }) {
  if (clip.cover_url) return <img className={className} src={clip.cover_url} alt={alt} draggable="false" />;
  // #t=0.1 makes iOS Safari paint a frame instead of a blank box
  if (clip.video_url) return <video className={className} src={`${clip.video_url}#t=0.1`} muted playsInline preload="metadata" aria-label={alt} />;
  return <div className={className} role="img" aria-label={alt} />;
}
