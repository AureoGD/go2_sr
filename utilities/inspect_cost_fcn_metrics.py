import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#TUDO: solve this import
# from environment.strategies.rgc_mpc.base_controller import LogCompletionDetector

WINDOW_SZ = 30
THD = 0.35
task1_extended = [
    -19.86313407912124, -16.740542859821304, -14.157607952298441, -12.030296276344222, -10.260823310532736,
    -8.796800716897167, -7.576367560535076, -6.561361448121159, -5.708978659941668, -4.973624486938868,
    -4.383740083856946, -3.897368814158623, -3.4926017416983575, -3.151122944348595, -2.861709068818965,
    -2.614347734254355, -2.401689073532555, -2.21805873431641, -2.056947734864748, -1.916316372389063,
    -1.7918968495889236, -1.6797884400679226, -1.578490970315708, -1.486245673372758, -1.4008463479052773,
    -1.3231123981454092, -1.2499490435185354, -1.1813605184951845, -1.1176780221899032, -1.0568137785925003,
    -0.9999192737793724, -0.9454683537971422, -0.8953020599453504, -0.8462408441801852, -0.8002831121710856,
    -0.7565842462714873, -0.7148233023667092, -0.6756079012879155, -0.6381114161708071, -0.6029537121980664,
    -0.5703440243577275, -0.5383593520124601, -0.5084419252726802, -0.4801342358526023, -0.4535155509663108,
    -0.42815140803328583, -0.40471582238622605, -0.3824579047971935, -0.36145575189195683, -0.34114264847934095,
    -0.3226612938062421, -0.3044571005138431, -0.2876127334243499, -0.2709788745094085, -0.2546707802255808,
    -0.23890800799330494, -0.22509256928928184, -0.212697751520433, -0.2012792790158052, -0.19023277059328866,
    -0.1798999063961936, -0.17025840208298793, -0.16096069014688985, -0.1523522106216301, -0.14420712678055722,
    -0.13661855177734733, -0.12947727613511706, -0.12273333161912289, -0.11648004705661817, -0.11059623754768606,
    -0.1051351863166824, -0.10017945738674201, -0.09593741214903206, -0.09174441713156586, -0.08760472731335975,
    -0.08362981878709823, -0.07993433306268873, -0.0764831380857171, -0.07330621823307502, -0.07038479382165445,
    -0.06763742599886247, -0.06511781686112322, -0.06269778132854317, -0.06050686716488568, -0.05844551546251605,
    -0.05649681509589611, -0.05464443133883853, -0.05288108357428027, -0.05119838780557893, -0.04958192761813027,
    -0.048058964089207803, -0.04660918327668187, -0.04521514112992261, -0.04388751918627533, -0.04260055783368325,
    -0.041363047628103874, -0.04017811106494933, -0.03904211847878575, -0.03796131240318277, -0.036892802333979555,
    -0.035869441930853826, -0.03486638752726576, -0.033915516006790714, -0.032975176073272494, -0.03209196493987257,
    -0.03122519910469005, -0.0303842003082986, -0.029568531279213685, -0.028780251441762796, -0.028015540041166884,
    -0.027272980631029023, -0.02657682367136835, -0.02588542771025261, -0.025211574500043667, -0.02455499561681709,
    -0.02391504033183816, -0.023295229954360393, -0.022692018795880762, -0.022140918006654364, -0.0215766398920793,
    -0.02103090363738381, -0.019720126134226277, -0.016661639908436155, -0.009871001679196844, 0.012615410170204285,
    0.9044076397173068, 0.5213733523708944, -0.04251903016715714, 0.23408601565280043, 0.002239273075223449,
    0.25530057058195793, -0.03112861445754022, 0.4926643796580202, 0.3509278777674085, 1.2082205225183353,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518,
    1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518, 1.2773380901719518
]


class LogCompletionDetector:
    """
    Simple log-space completion detector.
    """

    def __init__(self, window_size=20, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.log_history = []

    def is_task_complete(self, current_obj_val):
        abs_val = abs(current_obj_val)
        epsilon = 1e-10
        current_log = np.log10(abs_val + epsilon)

        self.log_history.append(current_log)

        if len(self.log_history) > self.window_size:
            self.log_history.pop(0)

        if len(self.log_history) < self.window_size:
            return False

        recent_logs = np.array(self.log_history)
        log_range = np.max(recent_logs) - np.min(recent_logs)

        return log_range < self.threshold

    def reset(self):
        self.log_history = []


# Convert to numpy array
task1 = np.array(task1_extended)
abs_task1 = np.abs(task1)
task1_log = np.log10(abs_task1 + 1e-10)

# Create comprehensive analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('LogCompletionDetector Analysis on NEW Task 1 Data', fontsize=16, fontweight='bold')

# Plot 1: Raw values (linear scale)
ax1 = axes[0, 0]
ax1.plot(task1, 'b-', linewidth=2, alpha=0.7)
ax1.set_xlabel('Step')
ax1.set_ylabel('Objective Value')
ax1.set_title('Raw Objective Values (78 steps)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 2: Absolute values with log y-axis
ax2 = axes[0, 1]
ax2.plot(abs_task1, 'g-', linewidth=2, alpha=0.7)
ax2.set_yscale('log')
ax2.set_xlabel('Step')
ax2.set_ylabel('|Value| (log scale)')
ax2.set_title('Absolute Values (Log Scale)')
ax2.grid(True, alpha=0.3)

# Plot 3: Log-transformed values
ax3 = axes[0, 2]
ax3.plot(task1_log, 'r-', linewidth=2, alpha=0.7)
ax3.set_xlabel('Step')
ax3.set_ylabel('log10(|Value|)')
ax3.set_title('Log-Transformed Values')
ax3.grid(True, alpha=0.3)

# Test different window sizes and thresholds
window_sizes = [5, 10, 15, 20]
thresholds = [0.05, 0.1, 0.15, 0.2]

# Create detector and analyze convergence
completion_data = {}

for ws in window_sizes:
    for th in thresholds:
        detector = LogCompletionDetector(window_size=ws, threshold=th)
        completion_steps = []

        for i, val in enumerate(task1):
            if detector.is_task_complete(val):
                completion_steps.append(i)

        key = f"ws{ws}_th{th}"
        completion_data[key] = {
            'window_size': ws,
            'threshold': th,
            'completion_steps': completion_steps,
            'first_completion': completion_steps[0] if completion_steps else None
        }

# Plot 4: Completion steps for different parameters
ax4 = axes[1, 0]
colors = ['blue', 'green', 'red', 'orange']
markers = ['o', 's', '^', 'D']

for idx, ws in enumerate(window_sizes):
    x_vals = []
    y_vals = []
    labels = []

    for th in thresholds:
        key = f"ws{ws}_th{th}"
        data = completion_data[key]
        if data['first_completion'] is not None:
            x_vals.append(th)
            y_vals.append(data['first_completion'])
            labels.append(f"th={th}")

    if x_vals:
        ax4.plot(x_vals, y_vals, marker=markers[idx], color=colors[idx], linewidth=2, markersize=8, label=f"WS={ws}")
        for x, y, label in zip(x_vals, y_vals, labels):
            ax4.text(x, y, label, fontsize=8, ha='center', va='bottom')

ax4.set_xlabel('Threshold')
ax4.set_ylabel('First Completion Step')
ax4.set_title('First Completion vs Threshold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Log range evolution for a specific detector configuration
ax5 = axes[1, 1]
# Test with window_size=10, threshold=0.1 (common starting point)
detector = LogCompletionDetector(window_size=WINDOW_SZ, threshold=THD)
log_ranges = []
convergence_flags = []

for i, val in enumerate(task1):
    # Get log range before checking completion
    if len(detector.log_history) >= 2:
        recent_logs = np.array(detector.log_history)
        log_range = np.max(recent_logs) - np.min(recent_logs)
        log_ranges.append(log_range)
    else:
        log_ranges.append(None)

    # Check completion
    is_complete = detector.is_task_complete(val)
    convergence_flags.append(is_complete)

# Plot log ranges - FIXED: Handle variable data length
valid_ranges = [r for r in log_ranges if r is not None]
if valid_ranges:  # Check if we have valid ranges
    steps = range(len(valid_ranges))
    ax5.plot(steps, valid_ranges, 'b-', linewidth=2, alpha=0.7, label='Log Range')
    ax5.axhline(y=float(THD), color='orange', linestyle='--', linewidth=2, label=f'Threshold {THD}')
    ax5.fill_between(steps, 0, 0.1, alpha=0.1, color='green')

    # Mark completion points - FIXED: Only plot if within range
    completion_steps = [i for i, flag in enumerate(convergence_flags) if flag]
    for cs in completion_steps:
        # Convert to index in valid_ranges (accounting for missing early values)
        cs_in_valid = cs - (len(log_ranges) - len(valid_ranges))
        if 0 <= cs_in_valid < len(valid_ranges):
            ax5.axvline(x=cs_in_valid, color='green', linestyle=':', alpha=0.7, linewidth=1)
            ax5.plot(cs_in_valid, valid_ranges[cs_in_valid], 'go', markersize=6)

ax5.set_xlabel('Step (offset for valid ranges)')
ax5.set_ylabel('Log Range (max - min)')
ax5.set_title(f'Log Range Evolution (WS={WINDOW_SZ}, TH={THD})')
if valid_ranges:  # Only add legend if we plotted something
    ax5.legend()
ax5.grid(True, alpha=0.3)
if valid_ranges:
    ax5.set_ylim(0, max(valid_ranges) * 1.1)

# Plot 6: Summary of all detectors' first completion
ax6 = axes[1, 2]

# Prepare data for heatmap-style visualization
heatmap_data = []
for ws in window_sizes:
    row = []
    for th in thresholds:
        key = f"ws{ws}_th{th}"
        data = completion_data[key]
        # Use -1 to indicate no completion for visualization
        row.append(data['first_completion'] if data['first_completion'] is not None else -1)
    heatmap_data.append(row)

# Create text annotations
for i, ws in enumerate(window_sizes):
    for j, th in enumerate(thresholds):
        value = heatmap_data[i][j]
        if value > 0:  # Only annotate if completion occurred
            ax6.text(j,
                     i,
                     f'{value}',
                     ha='center',
                     va='center',
                     color='white' if value > 40 else 'black',
                     fontweight='bold')
        elif value == -1:  # No completion
            ax6.text(j, i, 'NC', ha='center', va='center', color='gray', fontweight='bold')

# Create heatmap (replace -1 with 0 for visualization)
heatmap_vis = [[val if val > 0 else 0 for val in row] for row in heatmap_data]
im = ax6.imshow(heatmap_vis, cmap='YlOrRd', aspect='auto', vmin=0, vmax=78)

# Set labels
ax6.set_xticks(range(len(thresholds)))
ax6.set_xticklabels([f'{th}' for th in thresholds])
ax6.set_yticks(range(len(window_sizes)))
ax6.set_yticklabels([f'{ws}' for ws in window_sizes])

ax6.set_xlabel('Threshold')
ax6.set_ylabel('Window Size')
ax6.set_title('First Completion Step Heatmap (NC=No Completion)')
plt.colorbar(im, ax=ax6, label='First Completion Step')

plt.tight_layout()
plt.show()

# Print detailed analysis
print("=" * 80)
print("LOGCOMPLETIONDETECTOR ANALYSIS RESULTS - NEW DATA")
print("=" * 80)
print(f"Total steps analyzed: {len(task1)}")
print(f"Data range: {np.min(task1):.6f} to {np.max(task1):.6f}")
print(f"Absolute value range: {np.min(abs_task1):.6e} to {np.max(abs_task1):.6e}")
print(f"Log value range: {np.min(task1_log):.2f} to {np.max(task1_log):.2f}")
print()

print("First completion steps for each configuration:")
print("-" * 70)
print("Window Size | Threshold | First Completion | Value at Completion")
print("-" * 70)

for ws in window_sizes:
    for th in thresholds:
        key = f"ws{ws}_th{th}"
        data = completion_data[key]
        if data['first_completion'] is not None:
            step = data['first_completion']
            value = task1[step]
            print(f"{ws:11d} | {th:9.3f} | {step:16d} | {value:18.6e}")
        else:
            print(f"{ws:11d} | {th:9.3f} | {'No completion':16} | {'N/A':18}")

print()
print("=" * 80)
print("RECOMMENDATIONS FOR NEW DATA")
print("=" * 80)

# Analyze the best configuration
best_config = None
best_score = float('inf')

for ws in window_sizes:
    for th in thresholds:
        key = f"ws{ws}_th{th}"
        data = completion_data[key]
        if data['first_completion'] is not None:
            step = data['first_completion']
            # Score: earlier is better, but we want reasonable convergence
            # Penalize too early completion (before step 20)
            penalty = max(0, 20 - step) * 10
            score = step + penalty

            if score < best_score:
                best_score = score
                best_config = {'window_size': ws, 'threshold': th, 'step': step, 'value': task1[step], 'score': score}

if best_config:
    print(f"Recommended configuration:")
    print(f"  Window Size: {best_config['window_size']}")
    print(f"  Threshold: {best_config['threshold']}")
    print(f"  First detection at step: {best_config['step']}")
    print(f"  Value at detection: {best_config['value']:.6e}")
    print(f"  Log value: {np.log10(abs(best_config['value']) + 1e-10):.3f}")
    print()
    print(f"What this means:")
    print(f"  - Window of {best_config['window_size']} steps")
    print(f"  - Values vary by less than 10^{best_config['threshold']:.3f} = {10**best_config['threshold']:.3f}x")
    print(f"  - Max/min ratio < {10**best_config['threshold']:.3f}")
    print(f"  - At detection: |value| = {abs(best_config['value']):.6e}")
else:
    print("No convergence detected with any configuration!")

# Additional analysis: Show convergence progression
print()
print("=" * 80)
print("CONVERGENCE PROGRESSION WITH WS=15, TH=0.1")
print("=" * 80)

detector_balanced = LogCompletionDetector(window_size=15, threshold=0.1)
completion_progression = []

for i, val in enumerate(task1):
    if detector_balanced.is_task_complete(val):
        completion_progression.append({
            'step': i,
            'value': val,
            'log_value': np.log10(abs(val) + 1e-10),
        })

if completion_progression:
    print(f"First completion at step: {completion_progression[0]['step']}")
    print(f"Value: {completion_progression[0]['value']:.6e}")
    print(f"Log value: {completion_progression[0]['log_value']:.3f}")

    print(f"\nSubsequent completions (sustained convergence):")
    for i, cp in enumerate(completion_progression[1:10]):  # Show next 9
        print(f"  Step {cp['step']:3d}: value={cp['value']:12.6e}, log={cp['log_value']:7.3f}")
else:
    print("No convergence detected with WS=15, TH=0.1")

# Create one more plot showing the convergence behavior
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Raw values with convergence points for WS=15
ax1.plot(task1, 'b-', linewidth=1.5, alpha=0.7, label='Objective Value')
ax1.set_xlabel('Step')
ax1.set_ylabel('Objective Value')
ax1.set_title('Convergence Detection with Different Thresholds (WS=15)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)

# Mark completion points for different thresholds with WS=15
colors = ['green', 'orange', 'red', 'purple']
for idx, th in enumerate([0.05, 0.1, 0.15, 0.2]):
    detector = LogCompletionDetector(window_size=15, threshold=th)
    completion_steps = []

    for i, val in enumerate(task1):
        if detector.is_task_complete(val):
            completion_steps.append(i)

    if completion_steps:
        step = completion_steps[0]
        ax1.plot(step, task1[step], 'o', color=colors[idx], markersize=8, label=f'TH={th}, step={step}')
        ax1.axvline(x=step, color=colors[idx], linestyle=':', alpha=0.5)

ax1.legend(loc='upper right', fontsize=9)

# Plot 2: Show how values evolve in log space
ax2.plot(task1_log, 'b-', linewidth=1.5, alpha=0.7, label='log10(|Value|)')
ax2.set_xlabel('Step')
ax2.set_ylabel('log10(|Value|)')
ax2.set_title('Log-Space Evolution of Objective Values')
ax2.grid(True, alpha=0.3)

# Mark interesting regions
interesting_steps = []
for i in range(1, len(task1_log)):
    # Mark where log values start to flatten
    if i > 10 and abs(task1_log[i] - task1_log[i - 1]) < 0.01:
        interesting_steps.append(i)

if interesting_steps:
    first_flat = interesting_steps[0]
    ax2.axvline(x=first_flat, color='green', linestyle='--', alpha=0.7, label=f'First flattening (step {first_flat})')
    ax2.plot(first_flat, task1_log[first_flat], 'go', markersize=8)

ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
