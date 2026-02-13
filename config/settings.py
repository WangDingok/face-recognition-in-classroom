import argparse
from typing import Literal
from dataclasses import dataclass


@dataclass
class RecognitionConfig:
    """
    Configuration for the face recognition pipeline.

    Attributes
    ----------
    frame_skip_tracking : int
        Number of frames skipped during tracking.
    frame_skip_recognition : int
        Number of frames skipped before each recognition run.
    norm_threshold : float
        Maximum L2 norm allowed for embeddings.
    sim_threshold : float
        Minimum similarity required to accept a recognition result.
    top_k : int
        Number of top candidates retrieved from FAISS or the search backend.
    label_strategy : {'soft', 'hard'}
        Strategy used to assign labels.
    majority_ratio : float
        Minimum ratio required to accept a label in soft mode.
    vote_sim_threshold : float
        Similarity threshold for counting a vote in hard mode.
    min_valid : int
        Minimum number of valid samples required in hard mode.
    score_strategy : {'mean', 'max'}
        Method for aggregating similarity scores.
    sim_margin_update : float
        Margin used to update similarity over time during tracking.
    """

    frame_skip_tracking: int = 1
    frame_skip_recognition: int = 5
    norm_threshold: float = 10.0
    sim_threshold: float = 0.6
    top_k: int = 200
    label_strategy: Literal['soft', 'hard'] = 'soft'
    majority_ratio: float = 0.5
    vote_sim_threshold: float = 0.5
    min_valid: int = 10
    score_strategy: Literal['mean', 'max'] = 'max'
    sim_margin_update: float = 0.1



def parse_args():
    parser = argparse.ArgumentParser(description="Face Recognition Processor Configuration")

    # System-level paths and runtime
    parser.add_argument('--input_video_path', type=str, default=None,
                        help='Path to the input video')
    parser.add_argument('--output_video_path', type=str, default=None,
                        help='Path to the output video')
    parser.add_argument('--face_detector_model_path', type=str, default=None,
                        help='Path to the face detector model (e.g., yolov8-face.pt)')
    parser.add_argument('--face_index_path', type=str, default=None,
                        help='Path to the face index file (e.g., .index or .faiss)')
    parser.add_argument('--face_label_path', type=str, default=None,
                        help='Path to the face label file (e.g., .pkl)')
    parser.add_argument('--id_to_name_path', type=str, default=None,
                        help='Path to the ID-to-name dictionary in JSON format')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (e.g., "cuda" or "cpu")')

    # Recognition-related parameters
    parser.add_argument('--frame_skip_tracking', type=int, default=1,
                        help='Frames to skip for tracking')
    parser.add_argument('--frame_skip_recognition', type=int, default=5,
                        help='Frames to skip for recognition')
    parser.add_argument('--norm_threshold', type=float, default=21.0,
                        help='Embedding normalization threshold')
    parser.add_argument('--sim_threshold', type=float, default=0.6,
                        help='Similarity threshold used for recognition')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-K candidates to use for recognition')
    parser.add_argument('--label_strategy', type=str, choices=['soft', 'hard'], default='soft',
                        help='Strategy to assign labels: soft or hard voting')
    parser.add_argument('--majority_ratio', type=float, default=0.5,
                        help='Ratio required for a label to be accepted as majority')
    parser.add_argument('--vote_sim_threshold', type=float, default=0.5,
                        help='Similarity threshold used for voting in label_strategy (in hard mode)')
    parser.add_argument('--min_valid', type=int, default=10,
                        help='Minimum number of valid vectors to consider a match in label_strategy (in hard mode)')
    parser.add_argument('--score_strategy', type=str, choices=['mean', 'max'], default='max',
                        help='Aggregation method for similarity score')
    parser.add_argument('--sim_margin_update', type=float, default=0.1,
                        help='Similarity margin used when updating tracked identities')

    args = parser.parse_args()

    return args
