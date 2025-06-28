import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN


def interactive_drawing(width=512, height=512):
    canvas = np.zeros((height, width), dtype=np.uint8)
    drawing = False

    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            random_value = np.random.randint(0, 256)
            cv2.circle(canvas, (x, y), 5, int(random_value), -1)

    cv2.namedWindow("Draw your seeds (press s to save)")
    cv2.setMouseCallback("Draw your seeds (press s to save)", draw)

    while True:
        cv2.imshow("Draw your seeds (press s to save)", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
    cv2.destroyAllWindows()
    return canvas.astype(np.float32)

def simulate_blooming(image, exposure_time_s, rate_per_ms=1.0, timestep_ms=20,
                      ghost_seed_rate=0.01, ghost_neighbor_rate=0.1, ghost_threshold=50):
    
    time_ms = int(exposure_time_s * 1000)
    num_steps = time_ms // timestep_ms
    img = image.copy()
    h, w = img.shape
    frames = []

    pixel_nonuniformity = np.random.normal(loc=1.0, scale=0.05, size=img.shape).astype(np.float32)

    kernel = np.ones((3,3), dtype=np.float32)
    kernel[1,1] = 0

    vertical_kernel = np.zeros((3,3), dtype=np.float32)
    vertical_kernel[0,1] = 1
    vertical_kernel[2,1] = 1

    vertical_tail_probability = 0.3

    ghost_img = np.zeros_like(img)
    ghost_seed_points = []

    for step in range(num_steps):

        sources = img > 0
        img[sources] += rate_per_ms * timestep_ms
        img = np.clip(img, 0, 255)

        saturated = img >= 255

        if step == 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                saturated.astype(np.uint8), connectivity=8
            )
            if len(centroids) > 1:
                clustering = DBSCAN(eps=10, min_samples=1).fit(centroids[1:])
                for cluster_id in np.unique(clustering.labels_):
                    members = centroids[1:][clustering.labels_ == cluster_id]
                    avg_cx, avg_cy = np.mean(members, axis=0)
                    mirrored_x = w - int(avg_cx)
                    mirrored_y = h - int(avg_cy)
                    if 0 <= mirrored_x < w and 0 <= mirrored_y < h:
                        ghost_seed_points.append((mirrored_y, mirrored_x))
            else:
                cx, cy = centroids[0]
                mirrored_x = w - int(cx)
                mirrored_y = h - int(cy)
                if 0 <= mirrored_x < w and 0 <= mirrored_y < h:
                    ghost_seed_points.append((mirrored_y, mirrored_x))
            for gy, gx in ghost_seed_points:
                ghost_img[gy, gx] = 100  # reasonable seed

        # main bloom propagation
        neighbor_sum = cv2.filter2D(saturated.astype(np.float32), -1, kernel)
        normal_leak = neighbor_sum * (rate_per_ms * timestep_ms / 8)
        total_leak = normal_leak * np.random.normal(loc=1.0, scale=0.1, size=img.shape)

        if np.random.rand() < vertical_tail_probability:
            vertical_sum = cv2.filter2D(saturated.astype(np.float32), -1, vertical_kernel)
            vertical_leak = vertical_sum * (rate_per_ms * timestep_ms / 2)
            total_leak += vertical_leak * np.random.normal(loc=1.0, scale=0.1, size=img.shape)

        total_leak *= pixel_nonuniformity
        img += total_leak
        img = np.clip(img, 0, 255)

        # ghost bloom
        ghost_sources = ghost_img > 0
        ghost_img[ghost_sources] += rate_per_ms * ghost_seed_rate * timestep_ms

        ghost_spread = ghost_img >= ghost_threshold

        ghost_neighbor_sum = cv2.filter2D(ghost_spread.astype(np.float32), -1, kernel)
        ghost_normal_leak = ghost_neighbor_sum * (rate_per_ms * ghost_neighbor_rate * timestep_ms / 8)
        ghost_leak = ghost_normal_leak * np.random.normal(loc=1.0, scale=0.1, size=img.shape)

        if np.random.rand() < vertical_tail_probability:
            ghost_vertical_sum = cv2.filter2D(ghost_spread.astype(np.float32), -1, vertical_kernel)
            ghost_vertical_leak = ghost_vertical_sum * (rate_per_ms * ghost_neighbor_rate * timestep_ms / 2)
            ghost_leak += ghost_vertical_leak * np.random.normal(loc=1.0, scale=0.1, size=img.shape)

        ghost_leak *= pixel_nonuniformity
        ghost_img += ghost_leak
        ghost_img = np.clip(ghost_img, 0, 255)

        combined = img + ghost_img
        combined = np.clip(combined, 0, 255)

        if step % 5 == 0:
            frames.append(combined.astype(np.uint8))

    return frames

def animate_bloom(frames, interval_ms=50, exposure_time_s=2.0, save_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255)
    ax.axis("off")

    total_frames = len(frames)

    def update(i):
        frame = frames[i % total_frames]
        im.set_data(frame)
        current_time = (i % total_frames) * (exposure_time_s / total_frames)
        fig.suptitle(f"Exposure time: {current_time:.2f} s", fontsize=12)
        return [im]

    ani = FuncAnimation(fig, update, frames=np.arange(0, total_frames),
                        interval=interval_ms, repeat=True)

    if save_path:
        if not save_path.endswith(".gif"):
            save_path = save_path.rsplit(".", 1)[0] + ".gif"
        print(f"Saving to {save_path}...")
        ani.save(save_path, writer="pillow", fps=1000 // interval_ms)
        print("Saved!")

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Blooming with threshold and mirrored ghost artifact")
    parser.add_argument("--image", help="Path to image (optional)")
    parser.add_argument("--exposure", type=float, default=10.0, help="Exposure time in seconds")
    parser.add_argument("--speed", type=int, default=10, help="FPS")
    parser.add_argument("--stride", type=int, default=1, help="frame stride")
    parser.add_argument("--timestep", type=int, default=20, help="integration timestep ms")
    parser.add_argument("--threshold", type=int, default=20,
                        help="Ignore pixels below this value in loaded images (default 20)")
    parser.add_argument("--save", help="save path for gif")
    parser.add_argument("--ghost_seed_rate", type=float, default=0.01,
                        help="Ghost seed growth rate as fraction of rate_per_ms (default 0.01)")
    parser.add_argument("--ghost_neighbor_rate", type=float, default=0.1,
                        help="Ghost neighbor propagation rate as fraction of rate_per_ms (default 0.1)")
    parser.add_argument("--ghost_threshold", type=float, default=50,
                        help="Ghost propagation threshold (default 50)")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"cannot load {args.image}")
        img = img.astype(np.float32)
        img[img < args.threshold] = 0
    else:
        img = interactive_drawing()

    if img is None:
        print("no drawing provided, exiting.")
        exit()

    bloom_frames = simulate_blooming(
        img,
        args.exposure,
        rate_per_ms=1.0,
        timestep_ms=args.timestep,
        ghost_seed_rate=args.ghost_seed_rate,
        ghost_neighbor_rate=args.ghost_neighbor_rate,
        ghost_threshold=args.ghost_threshold
    )

    bloom_frames = bloom_frames[::args.stride]
    interval_ms = int(1000 / args.speed)

    animate_bloom(
        bloom_frames,
        interval_ms=interval_ms,
        exposure_time_s=args.exposure,
        save_path=args.save,
    )
