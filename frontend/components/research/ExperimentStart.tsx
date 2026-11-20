import Link from "next/link";

export function ExperimentStart() {
  return (
    <section className="experiment-start" aria-labelledby="start-title">
      <div className="start-copy">
        <p className="eyebrow">No research runs loaded</p>
        <h2 id="start-title">Explore a blind replay</h2>
        <p>
          See the inputs available to the model, make a prediction, then reveal
          the next session.
        </p>
        <div className="bench-actions">
          <Link href="/replay" className="bench-button primary-action">
            Watch AI competition <span aria-hidden="true">→</span>
          </Link>
          <Link href="/comparison" className="start-secondary">
            Open blind comparison
          </Link>
          <Link href="/studies?new=1" className="start-secondary">
            Train a model <span aria-hidden="true">↗</span>
          </Link>
        </div>
      </div>
      <figure
        className="window-map"
        aria-label="The model sees 64 previous returns. The next session is hidden until reveal."
      >
        <figcaption>Decision window</figcaption>
        <div className="window-diagram" aria-hidden="true">
          <div className="window-history">
            <strong>
              64 <small>past returns</small>
            </strong>
            <div className="return-cells" />
            <div className="window-offsets">
              <span>t − 63</span>
              <span>t</span>
            </div>
          </div>
          <div className="window-future">
            <span>?</span>t + 1
          </div>
        </div>
      </figure>
    </section>
  );
}
