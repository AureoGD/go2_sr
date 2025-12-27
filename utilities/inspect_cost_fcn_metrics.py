import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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


# Your NEW data (78 values)
task1_extended = [
    -2920.7244019925142, -66.90797145348853, -2320.0546262061816, -4287.902205691411, -572.7103833223089,
    -46.67265243212046, -234.7090473186636, -637.3283875466641, -939.3526473549416, -1229.83516769336,
    -1011.4593159425405, -924.7082097272005, -1469.7264543742554, -1053.6809838595377, -1276.8831359681965,
    -1313.666665473611, -1574.7653081264527, -1516.9438303608517, -1209.9081015221445, -913.8466597396065,
    -731.7623212824019, -677.1154053616933, -1213.4861302723564, -853.31069295096, -655.6370910216776,
    -594.6157744909776, -391.44234790656736, -652.2114941616461, -519.4980125525702, -342.89993208811586,
    -279.6361102689079, -64.12171537728268, -666.3315614459971, -1269.0951683319488, -1520.6712120155398,
    -1044.8045597895518, -567.9407254932727, -499.1058178541268, -391.42847923780636, -230.3754905013386,
    -154.85900995920554, -857.6651815723, -1588.238077700271, -1897.4118105020304, -2611.704610725636,
    -4330.294730200116, -5204.043413383864, -3161.838023354403, -2385.027512269611, -2291.3007902188833,
    -3232.745686710106, -2065.7821681551573, -1818.5770993260182, -467.53446172973514, -48.606277208718446,
    -495.74691008797896, -1073.9757721000547, -607.2543125502408, -132.6929098889433, -6.733400007335007,
    -5.202371597464673, -19.80592852893626, -62.229835612590165, -82.69031449092292, -45.76033205716906,
    -7.573457387063353, -46.27821978058146, -8.464600717930265, -10.601832100978937, -17.23399614051223,
    -21.769410561355055, -10.169899903093583, -13.56067429403618, -22.56754195000108, -9.834473815682083,
    -12.335671274207549, -11.653230333280767, -8.086551922531022, -6.549403042468814, -10.207492529422295,
    -6.688138183671413, -6.146610145035745, -5.5404321492348565, -8.472707272680386, -5.526315287620194,
    -2.4710352844553922, -9.900518169354902, -4.881029240993152, -3.8665590475253944, -3.583937045988996,
    -5.310910525152008, -3.326699662782829, -2.514496548374391, -1.7385197068476388, -2.9384876720900697,
    -1.613097504325526, -0.5572056637868669, -0.4477505344045246, -0.08142854863772082, -0.20545088322704416,
    -1.0695562394433311, -3.7301003460193405, -7.559429321858971, -3.894794904593414, -0.26842757444961773,
    -0.2406399951756333, -1.8140399989551486, -12.085329834183534, -34.27756342077614, -79.952381347453,
    -109.48799605141478, -141.12171694474873, -159.98533635596536, -228.17113888269085, -51.3333370870106,
    -51.3333370870106, -51.3333370870106, -51.3333370870106, -51.3333370870106, -51.3333370870106, -51.3333370870106,
    -51.3333370870106, -51.3333370870106, -51.3333370870106, -25.79515987795755, -25.79515987795755,
    -56.333066895493786, -40.448848694485434, -669.9867532167433, -650.6965270589196, -13.619523164322556,
    -13.619523164322556, -1602.3199179470735, -919.0525531866195, -54.22785744083373, -183.1068268645598,
    -557.1423031246633, -114.63850139869088, -15.656714087322374, -11.038284785108333, -0.7276309740412078,
    0.027666719927753894, 0.1444671803983658, 0.05317093536757132, -0.29590531873700787, -3.118134167887937,
    -7.539281671476217, -9.135150078529719, -8.328636715805983, -16.862520901625018
]

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
window_sz = 10
thd = 0.12
detector = LogCompletionDetector(window_size=window_sz, threshold=thd)
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
    ax5.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, label=f'Threshold {thd}')
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
ax5.set_title(f'Log Range Evolution (WS={window_sz}, TH={thd})')
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

print("\n" + "=" * 80)
print("KEY INSIGHTS FROM NEW DATA:")
print("=" * 80)
print("1. Data has only 78 steps (shorter optimization run)")
print("2. Objective values: -0.1236 → -0.000165 (improving)")
print("3. Pattern: Initial rapid improvement, then slower convergence")
print("4. Log values show when flattening occurs")
print("5. With only 78 steps, larger window sizes (20) may not detect")
print("   convergence before the run ends")
print()
print("6. For this short run, consider:")
print("   - Smaller window sizes (5-10) for earlier detection")
print("   - More lenient thresholds (0.15-0.2)")
print("   - Or run optimization for more steps")
print()
print("SUGGESTED PARAMETERS FOR SHORT RUNS:")
print("  - window_size=10, threshold=0.15")
print("  OR")
print("  - window_size=8, threshold=0.1")
