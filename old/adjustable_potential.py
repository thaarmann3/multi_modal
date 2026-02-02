#!/usr/bin/env python3
"""
Adjustable Potential Field with Voice Control
=============================================

A bare-bones structure that:
1. Creates a potential field with a single obstacle at the origin
2. Listens for spacebar press to trigger voice processing
3. Moves obstacle based on voice commands (left, right, forward, backward)
4. Visualizes before/after with movement vector
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_class import PotentialField
from voice_processing import capture_audio, transcribe_audio, pull_key_words


class AdjustablePotentialField:
    def __init__(self, x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0), 
                 resolution=200, scaling_factor=0.3):
        """Initialize the adjustable potential field."""
        self.pf = PotentialField(x_bounds=x_bounds, y_bounds=y_bounds, 
                                resolution=resolution, scaling_factor=scaling_factor)
        
        # Initial obstacle at origin
        self.obstacle_x = 0.0
        self.obstacle_y = 0.0
        self.obstacle_height = 40.0
        self.obstacle_width = 0.1
        self.move_step = 0.1  # Distance to move obstacle per command
        
        # Add initial obstacle
        self.pf.add_obstacle(self.obstacle_x, self.obstacle_y, 
                            self.obstacle_height, self.obstacle_width)
        
        # Store previous position and height for visualization
        self.prev_x = self.obstacle_x
        self.prev_y = self.obstacle_y
        self.prev_height = self.obstacle_height
        
        # Store previous potential field state for before/after comparison
        self.prev_pf = None
        self.prev_Z_pf = None
        
        # Height adjustment step
        self.height_step = 5.0  # Amount to increase/decrease height per command
        
        # Setup visualization with two subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.setup_plot()
        
        # Connect keyboard event
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def setup_plot(self):
        """Setup the initial plot."""
        # Calculate and display potential field
        Z_pf = self.pf.calculate_potential()
        # Store initial state as previous for first display
        self.prev_Z_pf = Z_pf.copy()
        
        # Left plot (before) - same as current for initial state
        self.ax1.clear()
        contour1 = self.ax1.contour(self.pf.X, self.pf.Y, Z_pf, levels=20, 
                                   alpha=0.6, cmap='viridis')
        self.ax1.clabel(contour1, inline=True, fontsize=8)
        self.ax1.plot(self.obstacle_x, self.obstacle_y, 'ro', 
                     markersize=12, label='Obstacle', zorder=5)
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_title('Before Movement')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.axis('equal')
        self.ax1.set_xlim(self.pf.x_bounds)
        self.ax1.set_ylim(self.pf.y_bounds)
        self.ax1.legend(loc='upper right')
        
        # Right plot (after) - same as current for initial state
        self.ax2.clear()
        contour2 = self.ax2.contour(self.pf.X, self.pf.Y, Z_pf, levels=20, 
                                   alpha=0.6, cmap='viridis')
        self.ax2.clabel(contour2, inline=True, fontsize=8)
        self.ax2.plot(self.obstacle_x, self.obstacle_y, 'ro', 
                     markersize=12, label='Obstacle', zorder=5)
        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Y (m)')
        self.ax2.set_title('After Movement - Press SPACEBAR to record voice command')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.axis('equal')
        self.ax2.set_xlim(self.pf.x_bounds)
        self.ax2.set_ylim(self.pf.y_bounds)
        self.ax2.legend(loc='upper right')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        
    def on_key_press(self, event):
        """Handle key press events."""
        if event.key == ' ':
            print("\nSpacebar pressed! Recording audio...")
            self.process_voice_command()
    
    def process_voice_command(self):
        """Process voice command and move obstacle."""
        try:
            # Record audio
            audio = capture_audio(duration=7.5)  # Record for 3 seconds
            
            # Transcribe
            transcription = transcribe_audio(audio)
            print(f"Transcription: {transcription}")
            
            # Parse command
            command = self.parse_command(transcription.lower())
            
            if command:
                if command in ['up', 'down']:
                    print(f"Adjusting obstacle height: {command}")
                    self.adjust_obstacle_height(command)
                else:
                    print(f"Moving obstacle: {command}")
                    self.move_obstacle(command)
            else:
                print("No valid command detected. Say 'left', 'right', 'forward', 'backward', 'up', or 'down'")
                
        except Exception as e:
            print(f"Error processing voice command: {e}")
    
    def parse_command(self, text):
        """Parse voice command from transcription, returning the most common keyword."""
        # Use pull_key_words to get keywords sorted by frequency
        # Set a high num_keywords to ensure we get all relevant words
        keywords, bigrams = pull_key_words(text, num_keywords=50, min_word_length=3)
        
        # Valid command keywords
        valid_commands = ['left', 'right', 'forward', 'backward', 'up', 'down']
        
        # Find the first (most common) keyword that matches a valid command
        for word, count in keywords:
            if word in valid_commands:
                return word
        
        return None
    
    def move_obstacle(self, direction):
        """Move obstacle in specified direction."""
        # Store previous position, height, and potential field state (before)
        self.prev_x = self.obstacle_x
        self.prev_y = self.obstacle_y
        self.prev_height = self.obstacle_height
        self.prev_Z_pf = self.pf.calculate_potential()
        
        # Calculate new position
        if direction == 'left':
            self.obstacle_x -= self.move_step
        elif direction == 'right':
            self.obstacle_x += self.move_step
        elif direction == 'forward':
            self.obstacle_y += self.move_step
        elif direction == 'backward':
            self.obstacle_y -= self.move_step
        
        # Keep obstacle within bounds
        self.obstacle_x = np.clip(self.obstacle_x, 
                                  self.pf.x_bounds[0] + 0.1, 
                                  self.pf.x_bounds[1] - 0.1)
        self.obstacle_y = np.clip(self.obstacle_y, 
                                  self.pf.y_bounds[0] + 0.1, 
                                  self.pf.y_bounds[1] - 0.1)
        
        # Update potential field
        self.pf.remove_obstacle(self.prev_x, self.prev_y)
        self.pf.add_obstacle(self.obstacle_x, self.obstacle_y, 
                            self.obstacle_height, self.obstacle_width)
        
        # Update visualization
        self.update_plot()
    
    def adjust_obstacle_height(self, direction):
        """Adjust obstacle height (magnitude) based on command."""
        # Store previous state
        self.prev_x = self.obstacle_x
        self.prev_y = self.obstacle_y
        self.prev_height = self.obstacle_height
        self.prev_Z_pf = self.pf.calculate_potential()
        
        # Adjust height
        if direction == 'up':
            self.obstacle_height += self.height_step
        elif direction == 'down':
            self.obstacle_height -= self.height_step
            # Prevent negative height
            self.obstacle_height = max(1.0, self.obstacle_height)
        
        # Update potential field with new height
        self.pf.remove_obstacle(self.obstacle_x, self.obstacle_y)
        self.pf.add_obstacle(self.obstacle_x, self.obstacle_y, 
                            self.obstacle_height, self.obstacle_width)
        
        # Update visualization
        self.update_plot()
    
    def update_plot(self):
        """Update plot showing before/after with movement vector."""
        # Calculate new potential field (after)
        Z_pf = self.pf.calculate_potential()
        
        # Left plot: Before movement
        self.ax1.clear()
        contour1 = self.ax1.contour(self.pf.X, self.pf.Y, self.prev_Z_pf, levels=20, 
                                    alpha=0.6, cmap='viridis')
        self.ax1.clabel(contour1, inline=True, fontsize=8)
        # Mark previous position (green circle)
        self.ax1.plot(self.prev_x, self.prev_y, 'go', 
                     markersize=12, label='Previous Position', zorder=5)
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_title('Before Movement')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.axis('equal')
        self.ax1.set_xlim(self.pf.x_bounds)
        self.ax1.set_ylim(self.pf.y_bounds)
        self.ax1.legend(loc='upper right')
        
        # Right plot: After movement
        self.ax2.clear()
        contour2 = self.ax2.contour(self.pf.X, self.pf.Y, Z_pf, levels=20, 
                                   alpha=0.6, cmap='viridis')
        self.ax2.clabel(contour2, inline=True, fontsize=8)
        
        # Mark previous position (green circle)
        self.ax2.plot(self.prev_x, self.prev_y, 'go', 
                     markersize=12, label='Previous Position', zorder=4)
        
        # Mark current position (red circle)
        self.ax2.plot(self.obstacle_x, self.obstacle_y, 'ro', 
                     markersize=12, label='Current Position', zorder=5)
        
        # Draw movement vector/arrow pointing from previous to current (tip at red)
        # Calculate movement direction (from previous to current)
        dx_total = self.obstacle_x - self.prev_x
        dy_total = self.obstacle_y - self.prev_y
        if abs(dx_total) > 1e-6 or abs(dy_total) > 1e-6:  # Only draw if there's actual movement
            # Calculate distance
            dist = np.sqrt(dx_total**2 + dy_total**2)
            
            # Estimate marker radius in data coordinates
            # markersize=12 corresponds to roughly 0.06 in data units for typical plot scales
            x_range = self.pf.x_bounds[1] - self.pf.x_bounds[0]
            marker_radius = 0.06 * x_range / 2.0  # Approximate radius
            head_length = 0.03  # Arrow head length (smaller for thinner arrow)
            
            # Normalize direction vector (pointing from previous to current)
            unit_dx = dx_total / dist
            unit_dy = dy_total / dist
            
            # Arrow should point from previous (green) to current (red)
            # Start from green circle edge (closest to red)
            start_x = self.prev_x + unit_dx * marker_radius
            start_y = self.prev_y + unit_dy * marker_radius
            
            # End point: red circle center (where tip should be)
            end_x = self.obstacle_x
            end_y = self.obstacle_y
            
            # Calculate arrow vector (pointing from previous to current)
            arrow_dx = end_x - start_x
            arrow_dy = end_y - start_y
            
            self.ax2.arrow(start_x, start_y, arrow_dx, arrow_dy, 
                          head_width=0.03, head_length=head_length, 
                          fc='blue', ec='blue', linewidth=1, 
                          label='Movement', zorder=6, length_includes_head=True)
        
        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Y (m)')
        self.ax2.set_title('After Movement - Press SPACEBAR to record voice command')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.axis('equal')
        self.ax2.set_xlim(self.pf.x_bounds)
        self.ax2.set_ylim(self.pf.y_bounds)
        self.ax2.legend(loc='upper right')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def run(self):
        """Run the interactive visualization."""
        print("=" * 60)
        print("Adjustable Potential Field with Voice Control")
        print("=" * 60)
        print("Instructions:")
        print("  - The plot window must be in focus for keyboard input")
        print("  - Press SPACEBAR to record a voice command")
        print("  - Say 'left', 'right', 'forward', or 'backward' to move the obstacle")
        print("  - Close the window to exit")
        print("=" * 60)
        
        plt.show(block=True)


if __name__ == "__main__":
    # Create and run the adjustable potential field
    apf = AdjustablePotentialField()
    apf.run()

