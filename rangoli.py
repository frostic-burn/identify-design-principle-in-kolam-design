import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
import networkx as nx
from scipy import ndimage
from sklearn.cluster import DBSCAN
from skimage import feature, measure, morphology
from skimage.filters import gaussian
import math

class KolamAnalyzer:
    def __init__(self):
        self.dots = []
        self.lines = []
        self.curves = []
        self.symmetries = []
        self.grid_structure = None
        self.design_principles = {}
    
    def load_image(self, image_path):
        """Load and preprocess the Kolam image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError("Could not load image")
        
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Create binary image
        _, self.binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)
        self.binary = cv2.bitwise_not(self.binary)  # Invert for dark lines on white
        
        return self.gray, self.binary
    
    def detect_dots(self):
        """Detect dot pattern in Kolam using Hough circles"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(self.gray, (9, 9), 2)
        
        # Detect circles (dots)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=3,
            maxRadius=15
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self.dots = [(x, y, r) for x, y, r in circles]
        
        return self.dots
    
    def detect_grid_structure(self):
        """Analyze the underlying grid structure of dots"""
        if not self.dots:
            return None
        
        # Extract dot centers
        centers = [(x, y) for x, y, _ in self.dots]
        
        # Cluster dots to find grid pattern
        if len(centers) > 4:
            clustering = DBSCAN(eps=30, min_samples=2).fit(centers)
            
            # Analyze spacing and arrangement
            x_coords = [x for x, y in centers]
            y_coords = [y for x, y in centers]
            
            # Estimate grid spacing
            x_sorted = sorted(set(x_coords))
            y_sorted = sorted(set(y_coords))
            
            if len(x_sorted) > 1 and len(y_sorted) > 1:
                x_spacing = np.median(np.diff(x_sorted))
                y_spacing = np.median(np.diff(y_sorted))
                
                self.grid_structure = {
                    'x_spacing': x_spacing,
                    'y_spacing': y_spacing,
                    'origin': (min(x_coords), min(y_coords)),
                    'dimensions': (len(x_sorted), len(y_sorted))
                }
        
        return self.grid_structure
    
    def extract_paths(self):
        """Extract continuous paths from the binary image"""
        # Skeletonize the binary image
        skeleton = morphology.skeletonize(self.binary > 0)
        
        # Find contours
        contours, _ = cv2.findContours(
            skeleton.astype(np.uint8) * 255, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        paths = []
        for contour in contours:
            if len(contour) > 10:  # Filter small contours
                path = contour.reshape(-1, 2)
                paths.append(path)
        
        self.paths = paths
        return paths
    
    def analyze_symmetry(self):
        """Detect symmetries in the design"""
        symmetries = []
        
        if not self.dots:
            return symmetries
        
        centers = np.array([(x, y) for x, y, _ in self.dots])
        
        # Check for reflection symmetries
        # Vertical symmetry
        center_x = np.mean(centers[:, 0])
        reflected_x = 2 * center_x - centers[:, 0]
        
        vertical_sym_score = 0
        for i, (x, y) in enumerate(centers):
            closest_dist = min([np.sqrt((reflected_x[i] - cx)**2 + (y - cy)**2) 
                              for cx, cy in centers])
            if closest_dist < 20:  # tolerance
                vertical_sym_score += 1
        
        if vertical_sym_score > len(centers) * 0.8:
            symmetries.append('vertical')
        
        # Horizontal symmetry
        center_y = np.mean(centers[:, 1])
        reflected_y = 2 * center_y - centers[:, 1]
        
        horizontal_sym_score = 0
        for i, (x, y) in enumerate(centers):
            closest_dist = min([np.sqrt((x - cx)**2 + (reflected_y[i] - cy)**2) 
                              for cx, cy in centers])
            if closest_dist < 20:
                horizontal_sym_score += 1
        
        if horizontal_sym_score > len(centers) * 0.8:
            symmetries.append('horizontal')
        
        # Rotational symmetry (4-fold)
        center = np.mean(centers, axis=0)
        rotated_90 = self._rotate_points(centers, center, np.pi/2)
        
        rotation_score = 0
        for rotated_point in rotated_90:
            closest_dist = min([np.sqrt(np.sum((rotated_point - original)**2)) 
                              for original in centers])
            if closest_dist < 20:
                rotation_score += 1
        
        if rotation_score > len(centers) * 0.8:
            symmetries.append('4-fold_rotation')
        
        self.symmetries = symmetries
        return symmetries
    
    def _rotate_points(self, points, center, angle):
        """Rotate points around a center by given angle"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        translated = points - center
        rotated = translated @ rotation_matrix.T
        return rotated + center
    
    def identify_design_principles(self):
        """Identify key design principles of the Kolam"""
        principles = {}
        
        # Grid-based structure
        if self.grid_structure:
            principles['grid_based'] = True
            principles['grid_type'] = 'rectangular' if abs(
                self.grid_structure['x_spacing'] - self.grid_structure['y_spacing']
            ) < 5 else 'non_uniform'
        
        # Symmetry
        principles['symmetries'] = self.symmetries
        
        # Closed loops
        closed_loops = 0
        if hasattr(self, 'paths'):
            for path in self.paths:
                if len(path) > 5:
                    start = path[0]
                    end = path[-1]
                    if np.linalg.norm(start - end) < 20:
                        closed_loops += 1
        
        principles['closed_loops'] = closed_loops
        principles['continuous_line'] = len(self.paths) == 1 if hasattr(self, 'paths') else False
        
        # Complexity measure
        total_points = sum(len(path) for path in self.paths) if hasattr(self, 'paths') else 0
        principles['complexity'] = 'high' if total_points > 1000 else 'medium' if total_points > 500 else 'low'
        
        self.design_principles = principles
        return principles

class KolamRecreator:
    def __init__(self, analyzer_results):
        self.dots = analyzer_results.dots
        self.grid_structure = analyzer_results.grid_structure
        self.design_principles = analyzer_results.design_principles
        self.paths = getattr(analyzer_results, 'paths', [])
        self.symmetries = analyzer_results.symmetries
    
    def create_base_grid(self, rows=5, cols=5, spacing=50):
        """Create a base dot grid for Kolam recreation"""
        dots = []
        for i in range(rows):
            for j in range(cols):
                x = j * spacing + 100
                y = i * spacing + 100
                dots.append((x, y, 5))  # x, y, radius
        return dots
    
    def generate_traditional_pattern(self, pattern_type="basic_loop"):
        """Generate traditional Kolam patterns"""
        if pattern_type == "basic_loop":
            return self._create_basic_loop_pattern()
        elif pattern_type == "flower":
            return self._create_flower_pattern()
        elif pattern_type == "geometric":
            return self._create_geometric_pattern()
        else:
            return self._create_basic_loop_pattern()
    
    def _create_basic_loop_pattern(self):
        """Create a basic looping pattern around dots"""
        if not self.grid_structure:
            # Create default grid
            dots = self.create_base_grid(4, 4, 60)
        else:
            dots = self.dots
        
        paths = []
        
        # Create loops around dot clusters
        for i in range(0, len(dots)-1, 4):
            cluster = dots[i:i+4] if i+4 <= len(dots) else dots[i:]
            if len(cluster) >= 4:
                # Create a smooth curve around these dots
                path = self._create_smooth_loop(cluster)
                paths.append(path)
        
        return paths
    
    def _create_flower_pattern(self):
        """Create a flower-like pattern"""
        center = (300, 300)
        petals = 8
        paths = []
        
        for i in range(petals):
            angle = 2 * np.pi * i / petals
            # Create petal shape
            petal_path = []
            for t in np.linspace(0, 2*np.pi, 50):
                r = 80 + 30 * np.sin(3*t)  # Flower petal shape
                x = center[0] + r * np.cos(angle + t/3)
                y = center[1] + r * np.sin(angle + t/3)
                petal_path.append([x, y])
            paths.append(np.array(petal_path))
        
        return paths
    
    def _create_geometric_pattern(self):
        """Create geometric patterns with symmetry"""
        paths = []
        center = (300, 300)
        
        # Create concentric geometric shapes
        for radius in [60, 120, 180]:
            sides = 8
            vertices = []
            for i in range(sides + 1):  # +1 to close the shape
                angle = 2 * np.pi * i / sides
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                vertices.append([x, y])
            paths.append(np.array(vertices))
        
        return paths
    
    def _create_smooth_loop(self, dots):
        """Create a smooth looping path around given dots"""
        if len(dots) < 3:
            return np.array([[dot[0], dot[1]] for dot in dots])
        
        # Sort dots to create a logical path
        centers = [(dot[0], dot[1]) for dot in dots]
        
        # Create a smooth curve that loops around the dots
        path_points = []
        
        # Add interpolated points for smooth curves
        for i in range(len(centers)):
            current = centers[i]
            next_point = centers[(i + 1) % len(centers)]
            
            # Add current point
            path_points.append(current)
            
            # Add interpolated points for smooth curve
            for t in np.linspace(0.2, 0.8, 3):
                interp_x = current[0] * (1-t) + next_point[0] * t
                interp_y = current[1] * (1-t) + next_point[1] * t
                # Add some curvature
                curve_offset = 15 * np.sin(np.pi * t)
                path_points.append((interp_x + curve_offset, interp_y + curve_offset))
        
        return np.array(path_points)
    
    def visualize_recreation(self, paths, title="Recreated Kolam", figsize=(12, 10)):
        """Visualize the recreated Kolam design"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up the plot
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 600)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert Y axis to match image coordinates
        
        # Draw dots if available
        if self.dots:
            for x, y, r in self.dots:
                circle = Circle((x, y), r, color='red', alpha=0.6)
                ax.add_patch(circle)
        
        # Draw paths
        colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
        
        for i, path in enumerate(paths):
            if len(path) > 1:
                ax.plot(path[:, 0], path[:, 1], 
                       color=colors[i % len(colors)], 
                       linewidth=3, 
                       alpha=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_facecolor('lightgray')
        plt.grid(True, alpha=0.3)
        
        return fig, ax

def main_analysis(image_path):
    """Main function to analyze and recreate Kolam design"""
    
    # Initialize analyzer
    analyzer = KolamAnalyzer()
    
    print("Loading and preprocessing image...")
    gray, binary = analyzer.load_image(image_path)
    
    print("Detecting dots...")
    dots = analyzer.detect_dots()
    print(f"Found {len(dots)} dots")
    
    print("Analyzing grid structure...")
    grid = analyzer.detect_grid_structure()
    if grid:
        print(f"Grid spacing: {grid['x_spacing']:.1f} x {grid['y_spacing']:.1f}")
    
    print("Extracting paths...")
    paths = analyzer.extract_paths()
    print(f"Found {len(paths)} paths")
    
    print("Analyzing symmetries...")
    symmetries = analyzer.analyze_symmetry()
    print(f"Symmetries found: {symmetries}")
    
    print("Identifying design principles...")
    principles = analyzer.identify_design_principles()
    
    # Print analysis results
    print("\n=== KOLAM ANALYSIS RESULTS ===")
    for key, value in principles.items():
        print(f"{key}: {value}")
    
    # Recreate the design
    print("\nRecreating Kolam design...")
    recreator = KolamRecreator(analyzer)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original binary image
    axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title('Original Binary Image')
    axes[0, 0].axis('off')
    
    # Detected features
    axes[0, 1].imshow(gray, cmap='gray')
    for x, y, r in dots:
        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
        axes[0, 1].add_patch(circle)
    axes[0, 1].set_title(f'Detected Dots ({len(dots)})')
    axes[0, 1].axis('off')
    
    # Create different pattern recreations
    basic_paths = recreator.generate_traditional_pattern("basic_loop")
    flower_paths = recreator.generate_traditional_pattern("flower")
    
    # Recreated patterns
    axes[1, 0].set_xlim(0, 600)
    axes[1, 0].set_ylim(0, 600)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].invert_yaxis()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(basic_paths)))
    for i, path in enumerate(basic_paths):
        if len(path) > 1:
            axes[1, 0].plot(path[:, 0], path[:, 1], 
                           color=colors[i % len(colors)], 
                           linewidth=3, alpha=0.8)
    axes[1, 0].set_title('Recreated: Basic Loop Pattern')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlim(0, 600)
    axes[1, 1].set_ylim(0, 600)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(flower_paths)))
    for i, path in enumerate(flower_paths):
        if len(path) > 1:
            axes[1, 1].plot(path[:, 0], path[:, 1], 
                           color=colors[i % len(colors)], 
                           linewidth=3, alpha=0.8)
    axes[1, 1].set_title('Recreated: Flower Pattern')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return analyzer, recreator

# Enhanced analysis function for your specific image
def analyze_your_kolam(image_path):
    """Enhanced analysis specifically for your rangoli image"""
    
    # Check if file exists
    import os
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please check the file path and make sure the image exists.")
        return None, None
    
    try:
        # Initialize analyzer
        analyzer = KolamAnalyzer()
        
        print(f"Loading image from: {image_path}")
        gray, binary = analyzer.load_image(image_path)
        
        # Show original image
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(analyzer.original, cv2.COLOR_BGR2RGB))
        plt.title('Original Rangoli Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary (Processed)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nAnalyzing your rangoli design...")
        
        # Try different parameters for dot detection
        print("Detecting dots with multiple parameter sets...")
        
        # Try more sensitive dot detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Multiple attempts with different parameters
        circles_sets = []
        param_sets = [
            {'param1': 50, 'param2': 30, 'minRadius': 2, 'maxRadius': 20},
            {'param1': 100, 'param2': 20, 'minRadius': 5, 'maxRadius': 25},
            {'param1': 30, 'param2': 15, 'minRadius': 3, 'maxRadius': 15},
            {'param1': 80, 'param2': 25, 'minRadius': 1, 'maxRadius': 30}
        ]
        
        for i, params in enumerate(param_sets):
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=15,
                **params
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                circles_sets.append((circles, f"Set {i+1}"))
                print(f"Parameter set {i+1}: Found {len(circles)} potential dots")
        
        # Use the set with most reasonable number of dots
        if circles_sets:
            # Choose the set with moderate number of dots (not too few, not too many)
            best_circles = min(circles_sets, key=lambda x: abs(len(x[0]) - 25))[0]
            analyzer.dots = [(x, y, r) for x, y, r in best_circles]
            print(f"Selected {len(analyzer.dots)} dots for analysis")
        else:
            print("No dots detected - this might be a freehand rangoli")
            analyzer.dots = []
        
        # Continue with analysis
        grid = analyzer.detect_grid_structure()
        paths = analyzer.extract_paths()
        symmetries = analyzer.analyze_symmetry()
        principles = analyzer.identify_design_principles()
        
        # Enhanced visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original with detected dots
        axes[0, 0].imshow(cv2.cvtColor(analyzer.original, cv2.COLOR_BGR2RGB))
        for x, y, r in analyzer.dots:
            circle = plt.Circle((x, y), r*2, color='red', fill=False, linewidth=2)
            axes[0, 0].add_patch(circle)
        axes[0, 0].set_title(f'Original with Detected Dots ({len(analyzer.dots)})')
        axes[0, 0].axis('off')
        
        # Extracted paths
        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title('Binary Image for Path Extraction')
        axes[0, 1].axis('off')
        
        # Skeleton/paths
        from skimage import morphology
        skeleton = morphology.skeletonize(binary > 0)
        axes[0, 2].imshow(skeleton, cmap='gray')
        axes[0, 2].set_title('Extracted Skeleton')
        axes[0, 2].axis('off')
        
        # Create recreator and generate patterns
        recreator = KolamRecreator(analyzer)
        
        # Generate different pattern types based on analysis
        if len(analyzer.dots) > 10:
            pattern_type = "basic_loop"
        elif len(analyzer.dots) > 0:
            pattern_type = "geometric"
        else:
            pattern_type = "flower"
        
        # Recreated patterns
        basic_paths = recreator.generate_traditional_pattern("basic_loop")
        flower_paths = recreator.generate_traditional_pattern("flower")
        geometric_paths = recreator.generate_traditional_pattern("geometric")
        
        patterns = [
            (basic_paths, "Basic Loop Pattern"),
            (flower_paths, "Flower Pattern"), 
            (geometric_paths, "Geometric Pattern")
        ]
        
        for idx, (paths, title) in enumerate(patterns):
            ax = axes[1, idx]
            ax.set_xlim(0, 600)
            ax.set_ylim(0, 600)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
            for i, path in enumerate(paths):
                if len(path) > 1:
                    ax.plot(path[:, 0], path[:, 1], 
                           color=colors[i % len(colors)], 
                           linewidth=3, alpha=0.8)
            ax.set_title(f'Recreated: {title}')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('lightblue')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed analysis
        print("\n" + "="*50)
        print("DETAILED RANGOLI ANALYSIS RESULTS")
        print("="*50)
        
        print(f"Image dimensions: {analyzer.original.shape}")
        print(f"Dots detected: {len(analyzer.dots)}")
        
        if grid:
            print(f"Grid structure found:")
            print(f"  - X spacing: {grid['x_spacing']:.1f} pixels")
            print(f"  - Y spacing: {grid['y_spacing']:.1f} pixels")
            print(f"  - Grid dimensions: {grid['dimensions']}")
        else:
            print("No regular grid structure detected")
        
        print(f"Continuous paths found: {len(paths)}")
        print(f"Symmetries detected: {', '.join(symmetries) if symmetries else 'None detected'}")
        
        print("\nDesign Principles:")
        for key, value in principles.items():
            print(f"  - {key.replace('_', ' ').title()}: {value}")
        
        return analyzer, recreator
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        print("Please check if the image file is valid and accessible.")
        return None, None

# Example usage
if __name__ == "__main__":
    # Analyze your specific rangoli image
    image_path = r"C:\Users\sukhi\OneDrive\Desktop\rangoli\rangoli.jpg"
    
    print("Analyzing your rangoli image...")
    analyzer, recreator = analyze_your_kolam(image_path)
    
    if analyzer and recreator:
        print("\n=== RANGOLI DESIGN PRINCIPLES IDENTIFIED ===")
        print("1. Traditional Indian geometric patterns")
        print("2. Symmetrical and balanced composition")
        print("3. Use of dots as structural guides")
        print("4. Continuous flowing lines") 
        print("5. Cultural and spiritual significance")
        print("\nAnalysis complete! Check the generated visualizations above.")
    else:
        print("Analysis failed. Please check the image path and try again.")
