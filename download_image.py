import os
import shutil
import multiprocessing
from bing_image_downloader import downloader

# --- Configuration ---
DATA_DIR = "real_world_data"
NUM_IMAGES_PER_CLASS = 60  # Total desired images for each category
HARD_TIMEOUT_PER_QUERY = 95  # Max seconds to spend on a single query process

# Define multiple, more descriptive search queries for each hazard class
HAZARD_QUERIES = {
    "pothole": ["road pothole", "street pothole damage", "asphalt deterioration road"],
    "sharp_turn": ["winding road", "sharp road curve", "hairpin turn road"],
    "object_on_road": ["debris on highway", "fallen rock on road", "obstacle on street"],
    "slippery_road": ["wet road surface", "icy road condition", "driving in heavy rain"],
    "landslide": ["landslide blocking road", "rockfall on highway", "mudslide across road"],
    "animal_crossing": ["deer crossing highway", "dog on road", "animal on road"],
    "normal": ["clear open road", "sunny day highway driving", "empty straight road"],
}


def download_worker(query, limit, output_dir):
    """A worker function to download images. Runs in a separate process."""
    try:
        downloader.download(
            query=query,
            limit=limit,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=10,  # Timeout for each individual image download
            verbose=True,  # Keep the downloader quiet; our script gives updates.
        )
    except Exception as e:
        print(f"\n[ERROR] Worker for query '{query}' failed: {e}")


def main():
    """
    Main function to orchestrate downloading and organizing images for all classes.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for hazard, queries in HAZARD_QUERIES.items():
        print(f"\n--- Processing Class: {hazard.upper()} ---")
        target_folder = os.path.join(DATA_DIR, hazard)
        os.makedirs(target_folder, exist_ok=True)

        # Count existing images to see if we need more
        image_counter = len(os.listdir(target_folder))
        if image_counter >= NUM_IMAGES_PER_CLASS:
            print(f"'{hazard}' already has {image_counter}/{NUM_IMAGES_PER_CLASS} images. Skipping.")
            continue

        print(f"Found {image_counter} existing images. Goal is {NUM_IMAGES_PER_CLASS}.")

        for query in queries:
            # Check if we've hit the goal before starting a new download
            if image_counter >= NUM_IMAGES_PER_CLASS:
                print(f"[INFO] Goal reached for '{hazard}'. Moving to the next class.")
                break

            # Calculate how many images to request, with a small buffer
            images_needed = NUM_IMAGES_PER_CLASS - image_counter
            limit_per_query = min(images_needed + 5, 20) # Request a few extra, cap at 20

            print(f"\nExecuting query: '{query}' (requesting ~{limit_per_query} images)")

            # --- Multiprocessing for Hard Timeout ---
            process = multiprocessing.Process(
                target=download_worker, args=(query, limit_per_query, DATA_DIR)
            )
            process.start()
            process.join(timeout=HARD_TIMEOUT_PER_QUERY)

            if process.is_alive():
                print(f"[TIMEOUT] Query '{query}' took more than {HARD_TIMEOUT_PER_QUERY}s. Terminating.")
                process.terminate()
                process.join()  # Ensure the process is fully cleaned up
                continue  # Skip to the next query

            # --- Process and Move Files ---
            print(f"Query '{query}' completed. Organizing files...")
            temp_folder = os.path.join(DATA_DIR, query)
            if not os.path.exists(temp_folder):
                print("[INFO] No new images were downloaded for this query.")
                continue

            moved_count = 0
            for filename in os.listdir(temp_folder):
                # Stop moving if we've hit the overall goal for the class
                if image_counter >= NUM_IMAGES_PER_CLASS:
                    break
                
                source_path = os.path.join(temp_folder, filename)
                if not os.path.isfile(source_path):
                    continue

                new_filename = f"{hazard}_{image_counter}.jpg"
                target_path = os.path.join(target_folder, new_filename)

                try:
                    shutil.move(source_path, target_path)
                    image_counter += 1
                    moved_count += 1
                except Exception as move_e:
                    print(f"[ERROR] Could not move file '{filename}': {move_e}")

            if moved_count > 0:
                print(f"Moved {moved_count} new images. Total for '{hazard}': {image_counter}/{NUM_IMAGES_PER_CLASS}.")

            # Clean up the temporary folder
            try:
                shutil.rmtree(temp_folder)
            except OSError as e:
                print(f"[ERROR] Could not remove temp folder '{temp_folder}': {e}")


if __name__ == "__main__":
    main()
    print("\n\n--- Image download process complete! ---")
    print(f"All images are saved in the '{DATA_DIR}' directory.")