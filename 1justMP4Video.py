import json
import urllib.request
import urllib.error
import urllib.parse
import time
import random
import uuid
import os
import copy
import subprocess # For running FFmpeg
# shutil might not be needed if not copying from a comfy output dir, but keep for now
import shutil   

# --- PyTorch and Dia Imports (with initial error handling) ---
TORCH_AVAILABLE = False
DIA_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    if not torch.cuda.is_available():
        print("Warning: PyTorch CUDA is not available. GPU operations will not be possible for Dia.")
except ImportError:
    print("Warning: PyTorch is not installed. Dia TTS might rely on it; GPU management features limited.")

try:
    from dia.model import Dia
    DIA_AVAILABLE = True
except ImportError:
    print("ERROR: Could not import Dia. Make sure the Dia library is installed.")
    print("You can typically install it with: pip install dia-tts")

# --- Configuration ---
# ComfyUI Settings
COMFYUI_SERVER_URL = "http://127.0.0.1:8188"
COMFYUI_CLIENT_ID = str(uuid.uuid4())
VIDEO_WORKFLOW_API_FILE = "wan2.1_t2v_workflow.json" # THIS MUST NOW CONTAIN YOUR MP4 SAVE NODE

# Content File
PREGENERATED_CONTENT_FILE = "pregenerated_content.json"

# Output Directories (created locally where the script runs)
BASE_PROJECT_DIR = os.getcwd() 
VIDEO_OUTPUTS_DIR = os.path.join(BASE_PROJECT_DIR, "video_outputs")
# VIDEO_WEBP_SUBDIR is no longer needed if MP4s are direct
VIDEO_MP4_SUBDIR = os.path.join(VIDEO_OUTPUTS_DIR, "mp4_clips") # Directly downloaded MP4 clips
SOUND_OUTPUTS_DIR = os.path.join(BASE_PROJECT_DIR, "sound_outputs")
FINAL_VIDEO_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "final_video_output")
LOGS_AND_MANIFESTS_DIR = os.path.join(BASE_PROJECT_DIR, "logs_and_manifests")

# Node IDs from your MODIFIED ComfyUI workflow
POSITIVE_PROMPT_NODE_ID = "6"
KSAMPLER_NODE_ID = "3"
EMPTY_LATENT_VIDEO_NODE_ID = "40"
# !!! IMPORTANT: Update this to the ID of your NEW MP4 Save Node in ComfyUI !!!
SAVE_MP4_NODE_ID = "52" # e.g., "35" or whatever it is

# Default Video Parameters
DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_FRAMES = 65 

# Dia TTS Settings
DIA_MODEL_NAME = "nari-labs/Dia-1.6B"
DIA_COMPUTE_DTYPE = "float16"
FULL_NARRATION_FILENAME = "full_narration.wav"

# Final Video Settings
FINAL_VIDEO_FILENAME = "final_narrative_video.mp4"
CONCATENATED_VIDEO_NO_AUDIO_FILENAME = "concatenated_video_no_audio.mp4" 
TEMP_EXTENDED_VIDEO_FILENAME = "temp_extended_video_for_mux.mp4" 


# --- Helper Functions ---

def create_output_dirs():
    # os.makedirs(VIDEO_WEBP_SUBDIR, exist_ok=True) # No longer needed
    os.makedirs(VIDEO_MP4_SUBDIR, exist_ok=True) # This will store downloaded MP4s
    os.makedirs(SOUND_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FINAL_VIDEO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_AND_MANIFESTS_DIR, exist_ok=True)
    print("Output directories ensured locally.")

def get_comfy_workflow_from_file(filepath):
    # ... (same as before)
    print(f"Attempting to load workflow from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: Workflow file not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        print(f"Workflow loaded successfully from {filepath}")
        return workflow
    except Exception as e:
        print(f"Error loading ComfyUI workflow from {filepath}: {e}")
        return None

def comfy_queue_prompt(prompt_workflow, client_id):
    # ... (same as before)
    try:
        p = {"prompt": prompt_workflow, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"{COMFYUI_SERVER_URL}/prompt", data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"ComfyUI HTTP Error queuing prompt: {e.code} {e.reason} {e.read().decode(errors='ignore')}")
        return None
    except Exception as e:
        print(f"ComfyUI Error queuing prompt: {e}")
        return None

def comfy_get_history(prompt_id):
    # ... (same as before)
    try:
        with urllib.request.urlopen(f"{COMFYUI_SERVER_URL}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except:
        return None

def download_comfy_output(filename, subfolder, file_type, target_local_path):
    # ... (same as before, this should work for MP4s too if served via /view)
    if not filename:
        print(f"  Error: Download attempt with no filename. Subfolder: '{subfolder}', Type: '{file_type}'.")
        return False

    data = {"filename": filename, "subfolder": subfolder, "type": file_type}
    url_values = urllib.parse.urlencode(data)
    view_url = f"{COMFYUI_SERVER_URL}/view?{url_values}"
    
    print(f"  Attempting to download from ComfyUI API: {view_url}")
    print(f"  Target local save path: {target_local_path}")

    try:
        os.makedirs(os.path.dirname(target_local_path), exist_ok=True)
        req = urllib.request.Request(view_url)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                with open(target_local_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"  SUCCESS: File downloaded and saved to {target_local_path}")
                if os.path.exists(target_local_path) and os.path.getsize(target_local_path) > 0:
                    print(f"  Verification: File exists and is not empty (size: {os.path.getsize(target_local_path)} bytes).")
                    return True
                else:
                    print(f"  ERROR: File saved but it's empty or missing post-save check. Path: {target_local_path}")
                    return False
            else:
                print(f"  ERROR: ComfyUI /view endpoint returned HTTP status {response.status}. Response: {response.read().decode(errors='ignore')}")
                return False
    except urllib.error.HTTPError as e:
        print(f"  HTTP ERROR during download: {e.code} {e.reason}")
        try: print(f"  Server response: {e.read().decode(errors='ignore')}")
        except: pass
        return False
    except Exception as e:
        print(f"  General ERROR during download or saving of {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comfyui_video_generation(video_prompt_text, base_workflow, output_filename_prefix,
                                 video_frames, video_width, video_height):
    if not base_workflow:
        return {"status": "ERROR_WORKFLOW_NOT_LOADED"} 

    workflow_to_run = copy.deepcopy(base_workflow)
    workflow_to_run[POSITIVE_PROMPT_NODE_ID]["inputs"]["text"] = video_prompt_text
    new_seed = random.randint(0, 2**32 - 1)
    workflow_to_run[KSAMPLER_NODE_ID]["inputs"]["seed"] = new_seed
    
    latent_inputs = workflow_to_run[EMPTY_LATENT_VIDEO_NODE_ID]["inputs"]
    latent_inputs["length"] = int(video_frames)
    latent_inputs["width"] = int(video_width)
    latent_inputs["height"] = int(video_height)

    # Update to use the new MP4 save node ID
    if SAVE_MP4_NODE_ID not in workflow_to_run:
        print(f"  FATAL ERROR in workflow: Save MP4 Node ID '{SAVE_MP4_NODE_ID}' not found!")
        print(f"  Please check VIDEO_WORKFLOW_API_FILE and the SAVE_MP4_NODE_ID variable.")
        return {"status": "ERROR_SAVE_MP4_NODE_MISSING_IN_WORKFLOW"}
    workflow_to_run[SAVE_MP4_NODE_ID]["inputs"]["filename_prefix"] = output_filename_prefix
    
    queued_data = comfy_queue_prompt(workflow_to_run, COMFYUI_CLIENT_ID)
    if not queued_data or "prompt_id" not in queued_data:
        return {"status": "ERROR_QUEUE_FAILED"}
    
    prompt_id = queued_data["prompt_id"]
    max_poll_attempts = 360 # Adjust as needed for your video lengths
    poll_interval = 10 # Increased polling interval slightly
    print(f"  ComfyUI prompt queued (ID: {prompt_id}). Polling every {poll_interval}s for max {max_poll_attempts*poll_interval}s...")

    for attempt in range(max_poll_attempts):
        time.sleep(poll_interval)
        history = comfy_get_history(prompt_id)
        
        if history and prompt_id in history:
            prompt_history = history[prompt_id]
            if "outputs" in prompt_history:
                outputs = prompt_history["outputs"]
                if SAVE_MP4_NODE_ID in outputs: # Check for the new MP4 save node
                    node_output_data = outputs[SAVE_MP4_NODE_ID]
                    
                    print(f"  DEBUG: Output data for Save MP4 node '{SAVE_MP4_NODE_ID}':")
                    print(f"  DEBUG: {json.dumps(node_output_data, indent=2)}") 

                    found_file_info = None
                    # Check common keys for video/file outputs.
                    # The 'Save Video (VHS)' node from VideoHelperSuite typically uses 'ui': {'filename': [...], 'text': [...]}
                    # and the actual output is often under a key like 'videos' or might be directly in node_output_data
                    # if the node itself is the direct output type.
                    if "videos" in node_output_data and len(node_output_data["videos"]) > 0:
                        print("  DEBUG: Found 'videos' key with content.")
                        found_file_info = node_output_data["videos"][0]
                    elif "files" in node_output_data and len(node_output_data["files"]) > 0:
                        print("  DEBUG: Found 'files' key with content.")
                        found_file_info = node_output_data["files"][0]
                    elif "uris" in node_output_data and len(node_output_data["uris"]) > 0:
                        # Some nodes might return URIs that include filename, subfolder, type
                        print("  DEBUG: Found 'uris' key. Assuming first URI contains file info.")
                        # This needs careful parsing if the URI itself isn't directly usable for /view
                        # For now, let's assume it might be a dict like other file_info objects
                        uri_data = node_output_data["uris"][0]
                        if isinstance(uri_data, str): # If it's just a string URI, we might need to parse it.
                             print(f"  DEBUG: URI is a string: {uri_data}. This might need special parsing for /view parameters.")
                             # Attempt to guess based on common patterns if it's a /view or /file style URI
                             # This is speculative and might need adjustment based on actual URI format
                             if "filename=" in uri_data:
                                 parsed_uri = urllib.parse.urlparse(uri_data)
                                 query_params = urllib.parse.parse_qs(parsed_uri.query)
                                 found_file_info = {
                                     "filename": query_params.get("filename", [None])[0],
                                     "subfolder": query_params.get("subfolder", [""])[0],
                                     "type": query_params.get("type", ["output"])[0]
                                 }
                                 print(f"  DEBUG: Attempted to parse URI into file_info: {found_file_info}")
                        elif isinstance(uri_data, dict): # If it's a dict, treat it like other file_info
                            found_file_info = uri_data


                    # Fallback to checking 'gifs' or 'images' if others fail, though less likely for MP4
                    elif "gifs" in node_output_data and len(node_output_data["gifs"]) > 0: # Less likely for MP4
                        print("  DEBUG: Found 'gifs' key with content (unexpected for MP4 but checking).")
                        found_file_info = node_output_data["gifs"][0]
                    elif "images" in node_output_data and len(node_output_data["images"]) > 0: # Less likely for MP4
                        print("  DEBUG: Found 'images' key with content (unexpected for MP4 but checking).")
                        found_file_info = node_output_data["images"][0]


                    if found_file_info and "filename" in found_file_info and found_file_info["filename"]:
                        print(f"  DEBUG: Successfully extracted file_info: {found_file_info}")
                        if "type" not in found_file_info or not found_file_info["type"]:
                            found_file_info["type"] = "output" 
                            print(f"  DEBUG: 'type' key missing or empty, defaulting to 'output'.")
                        if "subfolder" not in found_file_info: # Ensure subfolder is at least an empty string
                            found_file_info["subfolder"] = "" 
                            print(f"  DEBUG: 'subfolder' key missing, defaulting to empty string.")
                        return {"status": "SUCCESS", "data": found_file_info}
                    else:
                        if found_file_info:
                             print(f"  DEBUG: Potential key found, but 'filename' field missing or empty. Data: {found_file_info}")
                             return {"status": "ERROR_POTENTIAL_KEY_LACKS_FILENAME"}
                        else:
                             print("  DEBUG: No known/expected keys ('videos', 'files', 'uris', etc.) contained usable file information.")
                             return {"status": "ERROR_NO_USABLE_KEY_IN_OUTPUT_MP4"}
                else:
                    print(f"  DEBUG: Outputs from ComfyUI (node '{SAVE_MP4_NODE_ID}' not found as key):")
                    print(f"  DEBUG: {json.dumps(outputs, indent=2)}")
                    return {"status": "ERROR_SAVE_NODE_NOT_IN_OUTPUTS"}
            else: # Still processing
                if attempt % 6 == 0 : # Print status less frequently
                    status_str = prompt_history.get("status", {}).get("status_str", "Polling...")
                    q_rem = prompt_history.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                    print(f"  ComfyUI status: {status_str}, Queue remaining: {q_rem} (Poll {attempt+1}/{max_poll_attempts})")


    return {"status": "ERROR_TIMEOUT"}

def run_ffmpeg_command(command_list):
    # ... (same as before)
    print(f"Running FFmpeg command: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"FFmpeg Error (stdout): {stdout.decode(errors='ignore')}")
            print(f"FFmpeg Error (stderr): {stderr.decode(errors='ignore')}")
            return False
        return True
    except FileNotFoundError:
        print("FFmpeg Error: ffmpeg command not found. Ensure FFmpeg is in PATH.")
        return False
    except Exception as e:
        print(f"An exception occurred while running FFmpeg: {e}")
        return False

def get_media_duration(file_path):
    # ... (same as before)
    try:
        # Try to import ffmpeg-python only when needed
        # Ensure it's installed: pip install ffmpeg-python
        import ffmpeg 
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except ImportError:
        print("ffmpeg-python not installed. Falling back to subprocess ffprobe.")
    except Exception as e:
        print(f"Error getting duration for {file_path} with ffmpeg-python: {e}")
    
    # Fallback to subprocess ffprobe
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except FileNotFoundError:
        print(f"ffprobe command not found. Cannot get duration for {file_path}.")
        return None
    except Exception as sub_e:
        print(f"Subprocess ffprobe also failed for {file_path}: {sub_e}")
        return None

# --- Main Script Logic ---
if __name__ == "__main__":
    print("--- Starting Automated Narrative Generation (MP4 Direct Mode) ---")
    create_output_dirs()

    # --- Sanity Checks ---
    if SAVE_MP4_NODE_ID == "YOUR_NEW_MP4_SAVE_NODE_ID":
        print("FATAL ERROR: Please update the 'SAVE_MP4_NODE_ID' variable in the script with the actual node ID from your ComfyUI workflow!")
        exit()
    if not os.path.exists(VIDEO_WORKFLOW_API_FILE):
        print(f"FATAL ERROR: ComfyUI Workflow file not found: {VIDEO_WORKFLOW_API_FILE}")
        exit()
    if not os.path.exists(PREGENERATED_CONTENT_FILE):
        print(f"FATAL ERROR: Pregenerated content file not found: {PREGENERATED_CONTENT_FILE}")
        exit()
    if not DIA_AVAILABLE: # Dia check still relevant for Stage 4
        print(f"FATAL ERROR: Dia TTS library is not available. Install with: pip install dia-tts")
        exit()

    # === Stage 1: Load Content and ComfyUI Workflow ===
    print("\n--- Stage 1: Loading Content and Workflow ---")
    base_workflow = get_comfy_workflow_from_file(VIDEO_WORKFLOW_API_FILE)
    if not base_workflow: exit()
    try:
        with open(PREGENERATED_CONTENT_FILE, 'r', encoding='utf-8') as f:
            content_items = json.load(f)
        print(f"Loaded {len(content_items)} items from '{PREGENERATED_CONTENT_FILE}'.")
    except Exception as e:
        print(f"Error loading '{PREGENERATED_CONTENT_FILE}': {e}")
        exit()

    # === Stage 2: Generating Video Clips (ComfyUI) - Now expects MP4s ===
    print("\n--- Stage 2: Generating MP4 Video Clips with ComfyUI ---")
    clips_manifest = []
    comfyui_generation_overall_success = True

    for i, item in enumerate(content_items):
        item_id = item.get("id", i + 1)
        video_prompt = item.get("image_prompt") 
        
        if not video_prompt:
            print(f"Skipping item ID {item_id} (index {i}) - missing 'image_prompt'.")
            clips_manifest.append({"id": item_id, "clip_order": i, "status": "SKIPPED_NO_PROMPT", 
                                   "comfy_file_info": None, "mp4_path": None}) # Removed webp_path
            continue

        frames = item.get("duration_frames", DEFAULT_VIDEO_FRAMES)
        width = item.get("width", DEFAULT_VIDEO_WIDTH)
        height = item.get("height", DEFAULT_VIDEO_HEIGHT)
        safe_item_id_str = f"{item_id:04d}"
        output_prefix_for_comfy = f"narrativegen_clip_{safe_item_id_str}_"

        print(f"\nProcessing ComfyUI MP4 clip {i+1}/{len(content_items)} (ID: {item_id}, Prompt: '{video_prompt[:50]}...')")
        
        generation_result = run_comfyui_video_generation(
            video_prompt, base_workflow, output_prefix_for_comfy,
            frames, width, height
        )

        current_clip_status = generation_result["status"]
        comfy_file_info_dict = None
        local_mp4_storage_path = None # Changed from webp

        if current_clip_status == "SUCCESS":
            comfy_file_info_dict = generation_result.get("data")
            if comfy_file_info_dict and comfy_file_info_dict.get("filename"):
                retrieved_filename = comfy_file_info_dict["filename"]
                # Ensure the retrieved filename has .mp4 extension, or add it.
                # Some save nodes might not include it in the 'filename' field if they also set a 'format' field.
                if not retrieved_filename.lower().endswith(".mp4"):
                    print(f"  WARNING: ComfyUI filename '{retrieved_filename}' does not end with .mp4. Appending .mp4 for local save.")
                    retrieved_filename += ".mp4"
                
                local_mp4_storage_path = os.path.join(VIDEO_MP4_SUBDIR, retrieved_filename) # Save directly to mp4_subdir
                
                print(f"  ComfyUI reported success. File info: {comfy_file_info_dict}")
                if download_comfy_output(comfy_file_info_dict["filename"], 
                                         comfy_file_info_dict.get("subfolder", ""), 
                                         comfy_file_info_dict.get("type", "output"), 
                                         local_mp4_storage_path):
                    current_clip_status = "SUCCESS_MP4_DOWNLOADED" # Updated status
                else:
                    current_clip_status = "ERROR_MP4_DOWNLOAD_FAILED" # Updated status
                    local_mp4_storage_path = None 
                    comfyui_generation_overall_success = False
            else:
                current_clip_status = "ERROR_SUCCESS_NO_FILEDATA"
                print(f"  ComfyUI reported SUCCESS but API response missing file data: {generation_result}")
                comfyui_generation_overall_success = False
        else:
            print(f"  Failed ComfyUI generation for item ID {item_id}. Status: {current_clip_status}")
            comfyui_generation_overall_success = False

        clips_manifest.append({
            "id": item_id, "clip_order": i, "status": current_clip_status,
            "comfy_file_info": comfy_file_info_dict, 
            "mp4_path": local_mp4_storage_path # Store the MP4 path directly                     
        })

    manifest_path = os.path.join(LOGS_AND_MANIFESTS_DIR, "_generated_clips_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(clips_manifest, f, indent=2)
    print(f"\nMP4 video clip generation stage finished. Manifest: {manifest_path}")
    if not comfyui_generation_overall_success:
        print("WARNING: One or more MP4 video clips failed. Check logs and manifest.")

    # === Stage 3: Pausing for GPU Cooldown ===
    # ... (same as before) ...
    print("\n--- Stage 3: Pausing for 30 seconds (GPU cooldown) ---")
    time.sleep(30)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("Clearing PyTorch CUDA cache..."); torch.cuda.empty_cache(); print("CUDA cache cleared.")

    # === Stage 4: Generating Full Narration (Dia TTS) ===
    # ... (same as before) ...
    print("\n--- Stage 4: Generating Full Narration (Dia TTS) ---")
    full_narration_text = "".join([item.get("commentary", "") + " " for item in content_items if item.get("commentary")])
    narration_file_path = os.path.join(SOUND_OUTPUTS_DIR, FULL_NARRATION_FILENAME)

    if full_narration_text.strip():
        try:
            # To avoid Triton errors seen previously, consider use_torch_compile=False
            # Or ensure your PyTorch/Triton setup is fully compatible with your GPU for Dia.
            use_dia_torch_compile = True # Set to False if Triton errors persist with Dia
            print(f"Loading Dia TTS model (use_torch_compile={use_dia_torch_compile})..."); 
            dia_model = Dia.from_pretrained(DIA_MODEL_NAME, compute_dtype=DIA_COMPUTE_DTYPE)
            print(f"Generating narration ({len(full_narration_text)} chars)...")
            audio_output_dia = dia_model.generate(full_narration_text.strip(), use_torch_compile=use_dia_torch_compile, verbose=False)
            dia_model.save_audio(narration_file_path, audio_output_dia)
            print(f"Narration saved: {narration_file_path}")
            del dia_model
            if TORCH_AVAILABLE and torch.cuda.is_available(): torch.cuda.empty_cache()
            print("Dia model unloaded, CUDA cache cleared.")
        except Exception as e:
            print(f"ERROR generating Dia narration: {e}"); import traceback; traceback.print_exc(); narration_file_path = None 
    else:
        print("No commentary found. Skipping narration."); narration_file_path = None


    # === Stage 5: Pausing ===
    # ... (same as before) ...
    print("\n--- Stage 5: Pausing for 30 seconds ---")
    time.sleep(30)


    # === Stage 6: Assemble Final Video (FFmpeg) - SIMPLIFIED ===
    print("\n--- Stage 6: Assembling Final Video ---")
    
    # Step 6.1: MP4 conversion is NO LONGER NEEDED here.
    # We collect paths to already downloaded MP4s.
    valid_mp4_paths_for_concat = []
    clips_manifest.sort(key=lambda x: x.get("clip_order", float('inf'))) # Sort by order

    for clip_info in clips_manifest:
        if clip_info["status"] == "SUCCESS_MP4_DOWNLOADED" and clip_info["mp4_path"] and os.path.exists(clip_info["mp4_path"]):
            valid_mp4_paths_for_concat.append(clip_info["mp4_path"])
        else:
            print(f"Skipping clip ID {clip_info['id']} for concatenation due to status: {clip_info['status']} or missing MP4 path.")
            
    if not valid_mp4_paths_for_concat: 
        print("No valid MP4 clips to assemble. Exiting assembly stage.")
        # Ensure the script actually exits if this happens
        if __name__ == "__main__": # Simple way to ensure it exits if run as main
            exit()
    
    print(f"Collected {len(valid_mp4_paths_for_concat)} MP4 clips for concatenation.")

    # Step 6.2: Create filelist for FFmpeg concatenation
    # ... (same as before) ...
    filelist_path = os.path.join(LOGS_AND_MANIFESTS_DIR, "ffmpeg_filelist.txt")
    with open(filelist_path, 'w', encoding='utf-8') as fl:
        for mp4_p in valid_mp4_paths_for_concat: fl.write(f"file '{os.path.abspath(mp4_p)}'\n")
    print(f"FFmpeg filelist created: {filelist_path}")

    # Step 6.3: Concatenate MP4 clips
    # ... (same as before) ...
    concatenated_video_path = os.path.join(VIDEO_OUTPUTS_DIR, CONCATENATED_VIDEO_NO_AUDIO_FILENAME)
    concat_command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", filelist_path, "-c", "copy", "-y", concatenated_video_path]
    print("Concatenating MP4 clips...")
    if not run_ffmpeg_command(concat_command): 
        print("Failed to concatenate video clips. Cannot proceed with audio muxing.")
        if __name__ == "__main__": exit() 
    print(f"Concatenated video (no audio) saved: {concatenated_video_path}")


    # Step 6.4: Mux video with narration
    # ... (same as before, this logic should still work with the new concatenated_video_path) ...
    if narration_file_path and os.path.exists(narration_file_path):
        final_video_path = os.path.join(FINAL_VIDEO_OUTPUT_DIR, FINAL_VIDEO_FILENAME)
        
        video_duration = get_media_duration(concatenated_video_path)
        audio_duration = get_media_duration(narration_file_path)
        mux_command = None

        if video_duration is None or audio_duration is None:
            print("Could not get media durations. Attempting simple mux (-shortest).")
            mux_command = ["ffmpeg", "-i", concatenated_video_path, "-i", narration_file_path,
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-y", final_video_path]
        else:
            print(f"Concatenated video duration: {video_duration:.2f}s, Narration: {audio_duration:.2f}s")
            mux_video_input = concatenated_video_path
            temp_extended_video_path = os.path.join(VIDEO_OUTPUTS_DIR, TEMP_EXTENDED_VIDEO_FILENAME) 

            if audio_duration > video_duration:
                print("Audio is longer. Extending video's last frame...")
                padding_tpad_duration = audio_duration 
                extend_command = ["ffmpeg", "-i", concatenated_video_path, 
                                  "-vf", f"tpad=stop_mode=clone:stop_duration={padding_tpad_duration}", 
                                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", "-y", temp_extended_video_path]
                if run_ffmpeg_command(extend_command): 
                    mux_video_input = temp_extended_video_path
                    print(f"Temporarily extended video saved to: {mux_video_input}")
                else: 
                    print("Failed to extend video. Using original for muxing (result may be shorter than audio).")
            
            mux_command = ["ffmpeg", "-i", mux_video_input, 
                           "-i", narration_file_path, 
                           "-c:v", "copy", # Assumes mux_video_input is already well-encoded MP4
                           "-c:a", "aac", "-b:a", "192k", 
                           "-shortest",  
                           "-y", final_video_path]
        
        print("Muxing final video with narration...")
        if mux_command and run_ffmpeg_command(mux_command): 
            print(f"Final video created: {final_video_path}")
        else: 
            print(f"Failed to mux final video or no mux command was prepared.")
        
        if mux_video_input == temp_extended_video_path and os.path.exists(temp_extended_video_path):
            try: 
                os.remove(temp_extended_video_path)
                print(f"Removed temporary extended video: {temp_extended_video_path}")
            except Exception as e_rem: 
                print(f"Warning: Could not remove temporary file {temp_extended_video_path}: {e_rem}")
    else:
        print(f"Narration file not found/generated. Final video will not have custom audio.")
        print(f"The video with concatenated clips (no audio) is at: {concatenated_video_path}")
    
    print("\n--- Automated Narrative Generation Finished ---")