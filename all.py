MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
IMG_SIZE = 224

import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np  # Set the image size
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    
    preprocess_input = applications.inception_v3.preprocess_input

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return Model(inputs, outputs, name='feature_extractor')

# Build the feature extractor

def prepare_single_video(frames):
    feature_extractor = build_feature_extractor()
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype='bool')
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def sequence_prediction(path, model_path):
    model = tf.keras.models.load_model(model_path)
    frames = load_video(path)
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]

class DeepfakeVideoExplainability:
    def __init__(self, sequence_model, feature_extractor=None):
        self.sequence_model = sequence_model
        self.feature_extractor = feature_extractor or build_feature_extractor()
        self.class_names = ["REAL", 'FAKE']
        

    def feature_importance_analysis(self, video_path):
        frames = load_video(video_path)
        original_features, original_mask = prepare_single_video(frames)
        
        baseline_pred = self.sequence_model.predict([original_features, original_mask])[0][0]
        
        frame_importance = []
        valid_frames = min(len(frames), MAX_SEQ_LENGTH)
        
        for i in range(valid_frames):
            modified_features = original_features.copy()
            modified_features[0, i, :] = 0  
            
            modified_mask = original_mask.copy()
            modified_mask[0, i] = False  
            
            modified_pred = self.sequence_model.predict([modified_features, modified_mask])[0][0]
            
            importance = abs(baseline_pred - modified_pred)
            frame_importance.append(importance)
            
        while len(frame_importance) < MAX_SEQ_LENGTH:
            frame_importance.append(0.0)
        
        return {
            'frame_importance': np.array(frame_importance),
            'baseline_prediction': baseline_pred,
            'predicted_class': 'FAKE' if baseline_pred >= 0.5 else 'REAL',
            'confidence': baseline_pred if baseline_pred >= 0.5 else 1 - baseline_pred,
            'valid_frames': valid_frames
        }

    def temporal_attention_analysis(self, video_path):
        frames = load_video(video_path)
        frame_features, frame_mask = prepare_single_video(frames)
        
        # Get feature importance
        importance_result = self.feature_importance_analysis(video_path)
        frame_importance = importance_result["frame_importance"]
        
        # Calculate attention weights (softmax of importance scores)
        valid_importance = frame_importance[:importance_result["valid_frames"]]
        if np.sum(valid_importance) > 0:
            attention_weights = tf.nn.softmax(valid_importance).numpy()
        else:
            attention_weights = np.ones(len(valid_importance)) / len(valid_importance)
        
        # Pad with zeros
        full_attention = np.zeros(MAX_SEQ_LENGTH)
        full_attention[:len(attention_weights)] = attention_weights
        
        return {
            'attention_weights': full_attention,
            'peak_frames': np.argsort(attention_weights)[-3:][::-1],  # Top 3 frames
            'attention_entropy': -np.sum(attention_weights * np.log(attention_weights + 1e-10))
        }

    def feature_space_analysis(self, video_path):

        frames = load_video(video_path)
        frame_features, frame_mask = prepare_single_video(frames)
        
        # Get baseline prediction
        baseline_pred = self.sequence_model.predict([frame_features, frame_mask])[0][0]
        
        # Analyze feature importance by perturbation
        feature_importance = np.zeros(NUM_FEATURES)
        valid_frames = min(len(frames), MAX_SEQ_LENGTH)
        
        # Sample features to analyze (full analysis would be too slow)
        sample_features = np.random.choice(NUM_FEATURES, size=min(100, NUM_FEATURES), replace=False)
        
        for feat_idx in sample_features:
            # Zero out this feature across all frames
            modified_features = frame_features.copy()
            modified_features[0, :valid_frames, feat_idx] = 0
            
            # Get prediction
            modified_pred = self.sequence_model.predict([modified_features, frame_mask])[0][0]
            
            # Feature importance
            importance = abs(baseline_pred - modified_pred)
            feature_importance[feat_idx] = importance
        
        return {
            'feature_importance': feature_importance,
            'top_features': np.argsort(feature_importance)[-10:][::-1],
            'feature_stats': {
                'mean': np.mean(feature_importance),
                'std': np.std(feature_importance),
                'max': np.max(feature_importance)
            }
        }

    def spatial_attention_analysis(self, video_path):
        frames = load_video(video_path)
        
        # Ensure the feature extractor has been built by calling it once
        if not hasattr(self.feature_extractor, '_built') or not self.feature_extractor._built:
            # Build the model by calling it with a dummy input
            dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            _ = self.feature_extractor(dummy_input)
        
        # Get the InceptionV3 base model
        inception_base = None
        for layer in self.feature_extractor.layers:
            if hasattr(layer, 'name') and 'inception_v3' in layer.name.lower():
                inception_base = layer
                break
        
        # If we can't find InceptionV3 by name, try to find it by type
        if inception_base is None:
            for layer in self.feature_extractor.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 100:  # InceptionV3 has many layers
                    inception_base = layer
                    break
        
        # Fallback: use the last layer that has 'layers' attribute
        if inception_base is None:
            for layer in reversed(self.feature_extractor.layers):
                if hasattr(layer, 'layers'):
                    inception_base = layer
                    break
        
        if inception_base is None:
            print('Warning: Could not find InceptionV3 base model, using simplified spatial analysis')
            return self._simplified_spatial_analysis(video_path)
        
        # Find the last convolutional layer in InceptionV3
        last_conv_layer = None
        conv_layer_names = ["mixed10", 'mixed9', 'mixed8', 'mixed7']  # Common InceptionV3 layer names
        
        for layer_name in conv_layer_names:
            try:
                last_conv_layer = inception_base.get_layer(layer_name)
                break
            except ValueError:
                continue
        
        # If we can't find by name, find the last Conv2D layer
        if last_conv_layer is None:
            for layer in reversed(inception_base.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
        
        if last_conv_layer is None:
            print('Warning: Could not find convolutional layer, using simplified spatial analysis')
            return self._simplified_spatial_analysis(video_path)
        
        try:
            # Create GradCAM model
            grad_model = tf.keras.models.Model(
                inputs=self.feature_extractor.input,
                outputs=[last_conv_layer.output, self.feature_extractor.output]
            )
            
            spatial_maps = []
            valid_frames = min(len(frames), MAX_SEQ_LENGTH)
            
            for i in range(valid_frames):
                frame = frames[i:i+1].astype(np.float32)  # Single frame batch
                
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(frame)
                    # Use mean of features as proxy for 'importance'
                    target_output = tf.reduce_mean(predictions)
                
                # Get gradients
                grads = tape.gradient(target_output, conv_outputs)
                
                if grads is None:
                    # If gradients are None, create a dummy heatmap
                    heatmap = np.random.rand(8, 8)  # Approximate size
                else:
                    # Generate heatmap
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
                    heatmap = tf.maximum(heatmap, 0)
                    
                    # Normalize
                    if tf.reduce_max(heatmap) > 0:
                        heatmap = heatmap / tf.reduce_max(heatmap)
                    
                    heatmap = heatmap.numpy()
                
                spatial_maps.append(heatmap)
            
            return {
                'spatial_maps': spatial_maps,
                'num_frames': valid_frames
            }
            
        except Exception as e:
            print(f'Warning: GradCAM analysis failed ({e}), using simplified spatial analysis')
            return self._simplified_spatial_analysis(video_path)

    def _simplified_spatial_analysis(self, video_path):

        frames = load_video(video_path)
        valid_frames = min(len(frames), MAX_SEQ_LENGTH)
        
        # Create dummy spatial maps based on frame variance
        spatial_maps = []
        
        for i in range(valid_frames):
            frame = frames[i]
            
            # Calculate variance-based attention (areas with high variance get more attention)
            # Convert to grayscale for variance calculation
            gray_frame = np.mean(frame, axis=2)
            
            # Calculate local variance using a sliding window approach
            kernel_size = 16
            h, w = gray_frame.shape
            
            # Downsample to create a heatmap
            heatmap_h, heatmap_w = h // kernel_size, w // kernel_size
            heatmap = np.zeros((heatmap_h, heatmap_w))
            
            for y in range(heatmap_h):
                for x in range(heatmap_w):
                    y_start, y_end = y * kernel_size, (y + 1) * kernel_size
                    x_start, x_end = x * kernel_size, (x + 1) * kernel_size
                    
                    if y_end <= h and x_end <= w:
                        patch = gray_frame[y_start:y_end, x_start:x_end]
                        heatmap[y, x] = np.var(patch)
            
            # Normalize
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            spatial_maps.append(heatmap)
        
        return {
            'spatial_maps': spatial_maps,
            'num_frames': valid_frames
        }
    def comprehensive_analysis(self, video_path, output_dir=None):

        print(f'Analyzing video: {video_path}')
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Perform all analyses
        print('1. Feature importance analysis...')
        feature_analysis = self.feature_importance_analysis(video_path)
        
        print('2. Temporal attention analysis...')
        temporal_analysis = self.temporal_attention_analysis(video_path)
        
        print('3. Feature space analysis...')
        feature_space = self.feature_space_analysis(video_path)
        
        print('4. Spatial attention analysis...')
        spatial_analysis = self.spatial_attention_analysis(video_path)
        
        # Create comprehensive visualization
        self._create_comprehensive_visualization(
            feature_analysis, temporal_analysis, feature_space, spatial_analysis,
            video_path, output_dir
        )
        
        # Save detailed report
        if output_dir:
            self._save_detailed_report(
                feature_analysis, temporal_analysis, feature_space, spatial_analysis,
                video_path, output_dir
            )
        
        return {
            'feature_analysis': feature_analysis,
            'temporal_analysis': temporal_analysis,
            'feature_space': feature_space,
            'spatial_analysis': spatial_analysis
        }

    def _create_comprehensive_visualization(self, feature_analysis, temporal_analysis, 
                                         feature_space, spatial_analysis, video_path, output_dir):
      
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Frame importance
        valid_frames = feature_analysis["valid_frames"]
        frame_importance = feature_analysis["frame_importance"][:valid_frames]
        
        axes[0, 0].plot(range(valid_frames), frame_importance, 'b-', linewidth=2, marker='o')
        axes[0, 0].set_title('Frame Importance Analysis', fontsize=14)
        axes[0, 0].set_xlabel('Frame Index')
        axes[0, 0].set_ylabel('Importance Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Temporal attention
        attention_weights = temporal_analysis["attention_weights"][:valid_frames]
        
        axes[0, 1].bar(range(valid_frames), attention_weights, color='orange', alpha=0.7)
        axes[0, 1].set_title('Temporal Attention Weights', fontsize=14)
        axes[0, 1].set_xlabel('Frame Index')
        axes[0, 1].set_ylabel('Attention Weight')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance distribution
        feature_importance = feature_space["feature_importance"]
        non_zero_features = feature_importance[feature_importance > 0]
        
        if len(non_zero_features) > 0:
            axes[1, 0].hist(non_zero_features, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Feature Importance Distribution', fontsize=14)
            axes[1, 0].set_xlabel('Importance Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction summary
        axes[1, 1].text(0.1, 0.8, f'Video: {os.path.basename(video_path)}', fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.7, f'Prediction: {feature_analysis["predicted_class"]}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Confidence: {feature_analysis["confidence"]:.3f}', fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'Raw Score: {feature_analysis["baseline_prediction"]:.3f}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Valid Frames: {valid_frames}/{MAX_SEQ_LENGTH}', fontsize=12)
        
        # Peak frames
        peak_frames = temporal_analysis["peak_frames"]
        axes[1, 1].text(0.1, 0.3, f'Key Frames: {peak_frames[:3]}', fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'Attention Entropy: {temporal_analysis["attention_entropy"]:.3f}', fontsize=12)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Analysis Summary', fontsize=14)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()

    def _save_detailed_report(self, feature_analysis, temporal_analysis, feature_space, 
                            spatial_analysis, video_path, output_dir):
      
        report_file = os.path.join(output_dir, 'detailed_report.txt')
        
        with open(report_file, 'w') as f:
            f.write('DEEPFAKE DETECTION EXPLAINABILITY REPORT\n')
            f.write('=' * 50 + '\n\n')
            
            f.write(f'Video: {video_path}\n')
            f.write(f'Prediction: {feature_analysis["predicted_class"]}\n')
            f.write(f'Confidence: {feature_analysis["confidence"]:.3f}\n')
            f.write(f'Raw Score: {feature_analysis["baseline_prediction"]:.3f}\n')
            f.write(f'Valid Frames: {feature_analysis["valid_frames"]}/{MAX_SEQ_LENGTH}\n\n')
            
            f.write('TEMPORAL ANALYSIS:\n')
            f.write('-' * 30 + '\n')
            peak_frames = temporal_analysis["peak_frames"]
            attention_weights = temporal_analysis["attention_weights"]
            
            f.write('Most Important Frames:\n')
            for i, frame_idx in enumerate(peak_frames[:5]):
                f.write(f'  {i+1}. Frame {frame_idx}: {attention_weights[frame_idx]:.4f}\n')
            
            f.write(f'\nAttention Entropy: {temporal_analysis["attention_entropy"]:.3f}\n')
            f.write('(Lower entropy = more focused attention)\n\n')
            
            f.write('FEATURE SPACE ANALYSIS:\n')
            f.write('-' * 30 + '\n')
            f.write(f'Features analyzed: {np.sum(feature_space["feature_importance"] > 0)}\n')
            f.write(f'Mean feature importance: {feature_space["feature_stats"]["mean"]:.4f}\n')
            f.write(f'Max feature importance: {feature_space["feature_stats"]["max"]:.4f}\n')
            f.write(f'Feature importance std: {feature_space["feature_stats"]["std"]:.4f}\n\n')
            
            f.write('INTERPRETATION:\n')
            f.write('-' * 30 + '\n')
            
            if feature_analysis["predicted_class"] == 'FAKE':
                f.write('The model detected this video as FAKE.\n')
                f.write('Key indicators:\n')
                f.write(f'- High confidence score: {feature_analysis["confidence"]:.3f}\n')
            else:
                f.write('The model detected this video as REAL.\n')
                f.write('Key indicators:\n')
                f.write(f'- High confidence score: {feature_analysis["confidence"]:.3f}\n')
            
            if temporal_analysis["attention_entropy"] < 2.0:
                f.write('- Model shows focused attention (low entropy)\n')
            else:
                f.write('- Model shows distributed attention (high entropy)\n')
            
            f.write(f'- Analysis based on {feature_analysis["valid_frames"]} frames\n')
        
        print(f'Detailed report saved to: {report_file}')

    def create_overlay_video(self, video_path, spatial_analysis, output_path=None, fps=None, 
                            beyond_analysis_mode='last_map'):
        
        if not spatial_analysis["spatial_maps"]:
            print('No spatial maps available for overlay')
            return None
        
        # Get original video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Error: Could not open video {video_path}')
            return None
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS) if fps is None else fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f'Original video: {total_frames} frames at {original_fps} fps')
        
        # Load full video (not limited by MAX_SEQ_LENGTH)
        full_frames = load_video(video_path, max_frames=0)  # 0 means no limit
        cap.release()
        
        print(f'Loaded {len(full_frames)} frames for overlay')
        
        # Get spatial maps (these are limited to MAX_SEQ_LENGTH)
        spatial_maps = spatial_analysis["spatial_maps"]
        num_analysis_frames = len(spatial_maps)
        
        print(f'Available spatial maps: {num_analysis_frames}')
        print(f'Beyond analysis mode: {beyond_analysis_mode}')
        
        overlayed_frames = []
        
        for i, frame in enumerate(full_frames):
            # Convert frame to uint8
            frame_uint8 = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
            
            # Determine spatial map based on mode
            spatial_map = None
            overlay_strength = 0.3
            
            if i < num_analysis_frames:
                # Within analysis window
                spatial_map = spatial_maps[i]
                overlay_strength = 0.3
            else:
                # Beyond analysis window
                if beyond_analysis_mode == 'last_map':
                    spatial_map = spatial_maps[-1] if spatial_maps else None
                    overlay_strength = 0.15  # Reduced strength for beyond-analysis frames
                    
                elif beyond_analysis_mode == 'fade_out':
                    fade_frames = 30  # Fade out over 30 frames
                    frames_beyond = i - num_analysis_frames
                    if frames_beyond < fade_frames:
                        fade_factor = max(0, 1 - frames_beyond / fade_frames)
                        spatial_map = spatial_maps[-1] * fade_factor if spatial_maps else None
                        overlay_strength = 0.3 * fade_factor
                    else:
                        spatial_map = None
                        
                elif beyond_analysis_mode == 'repeat_analysis':
                    # Repeat the analysis pattern
                    pattern_idx = i % num_analysis_frames
                    spatial_map = spatial_maps[pattern_idx]
                    overlay_strength = 0.2  # Reduced strength for repeated pattern
                    
                elif beyond_analysis_mode == 'no_overlay':
                    spatial_map = None
            
            # Apply overlay if spatial map exists
            if spatial_map is not None:
                # Resize spatial map to match frame
                spatial_map_resized = cv2.resize(spatial_map, (IMG_SIZE, IMG_SIZE))
                
                # Create heatmap
                heatmap = cv2.applyColorMap(
                    (spatial_map_resized * 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                
                # Apply overlay
                overlay = cv2.addWeighted(frame_uint8, 0.7, heatmap, overlay_strength, 0)
                overlayed_frames.append(overlay)
            else:
                # No overlay, use original frame
                overlayed_frames.append(frame_uint8)
        
        if output_path:
            # Save as video with original fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, original_fps, (IMG_SIZE, IMG_SIZE))
            
            for frame in overlayed_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            print(f'Overlay video saved to: {output_path}')
            print(f'Duration: {len(overlayed_frames)} frames at {original_fps} fps = {len(overlayed_frames)/original_fps:.2f} seconds')
        
        return overlayed_frames

# Also update the explain_deepfake_video function to use the new options
def explain_deepfake_video(sequence_model_path, video_path, output_dir=None, create_overlay=True, 
                          overlay_mode='fade_out'):
    import os
    
    # Load model
    sequence_model = tf.keras.models.load_model(sequence_model_path)
    
    # Initialize explainer
    explainer = DeepfakeVideoExplainability(sequence_model)
    
    # Perform comprehensive analysis
    results = explainer.comprehensive_analysis(video_path, output_dir)
    
    # Create overlay video if requested and output directory is specified
    if create_overlay and output_dir:
        print('5. Creating overlay video...')
        
        # Generate output path for overlay video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        overlay_output_path = os.path.join(output_dir, f'{video_name}_overlay.mp4')
        
        # Create overlay video with specified mode
        overlay_frames = explainer.create_overlay_video(
            video_path, 
            results["spatial_analysis"], 
            overlay_output_path,
            beyond_analysis_mode=overlay_mode
        )
        
        # Add overlay information to results
        results["overlay_video_path"] = overlay_output_path
        results["overlay_frames_created"] = len(overlay_frames) if overlay_frames else 0
        
        print(f'Overlay video created with {results["overlay_frames_created"]} frames')
    
    elif create_overlay and not output_dir:
        print('Warning: Cannot create overlay video without output directory')
    
    return results


video_path = 'aassnaulhq.mp4'
model_path = 'video.keras'

if(sequence_prediction(video_path, model_path)>=0.5):
    print(f'The predicted class of the video is FAKE')
else:
    print(f'The predicted class of the video is REAL')

print('=================================================')

explain_deepfake_video(
    sequence_model_path=model_path, 
    video_path=video_path, 
    output_dir='/kaggle/working/', 
    create_overlay=True, 
    overlay_mode='repeat_analysis')

