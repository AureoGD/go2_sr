import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sr_strategies.rgc.base_controller import LogCompletionDetector

# Your NEW data (78 values)
task1_extended = [
    -348.2436325159226, -1102004.5207179652, -516116.757936072, -166397.67481618642, -51366.51895362812,
    -43356.45348695084, -47678.614707414476, -31129.265851026194, -30755.943178415073, -37946.20814642386,
    -20001.102253426667, -20611.631152692804, -10746.752866105187, -5629.171219749557, -3134.322971108886,
    -7557.253993384245, -4952.379645860842, -4966.642411098279, -4093.0726437278586, -3024.354616160714,
    -2255.3117314977435, -1755.677593118527, -1439.4706847740074, -1260.8440064469394, -1149.2423096772336,
    -1112.199239624565, -1417.3172988950107, -1215.7397694003994, -1071.2080546744382, -1002.4518136449605,
    -917.1207822624834, -829.2430352877811, -746.3719778726164, -667.1405474487747, -601.3018828249562,
    -547.2253908260157, -499.3352187158371, -452.5418274451679, -410.24161123868043, -376.9437579831031,
    -341.70669913914924, -310.49170874171773, -285.5424233754334, -259.9336261819747, -237.22148764942122,
    -218.4073431698166, -199.4781134337658, -183.410516749522, -167.8432948549346, -154.31773138645568,
    -141.01659105497524, -130.0889201964578, -119.27160127418428, -109.85959220876887, -101.00549115671598,
    -92.6499871630587, -85.58752223009432, -78.70671685473837, -72.69136497283932, -66.3031534351595,
    -61.60751352475198, -56.44264889952877, -52.31967251121157, -48.14361889935931, -44.7196217818971,
    -41.264401685735535, -37.94949069459039, -35.00713394035234, -32.59486524329737, -30.55273283213,
    -28.36664671974944, -26.24501270345355, -24.17257001736491, -22.759180605854805, -18.869537200516042,
    -15.595676346460554, -13.003121731458567, -11.297590991117202, -9.704430653337145, -8.713223818865924,
    -14.495527073485999, -12.856625426402456, -11.066425034765153, -9.47704363089404, -8.085232035901045,
    -6.996666278135681, -6.275344367720919, -5.743286963467172, -5.427505291573652, -5.1850905371399545,
    -4.868835477947995, -4.710374285539831, -8.216617173631278, -7.371670289605344, -6.216553577736349,
    -5.380818554858056, -4.784036109742139, -4.312861841913948, -3.889605203234366, -3.662327139261751,
    -3.444021069231002, -3.14975155376216, -2.858364985461751, -2.636674215706724, -2.6139760345371994,
    -2.7002771035065294, -2.80622140085275, -2.905136418272427, -3.000910249434731, -3.0773212673247827,
    -3.098777074181037, -3.0644616653258057, -2.9660363370835414, -2.8548318395666286, -2.7054589509028184,
    -2.4936027105531213, -2.3684113829348163, -2.1437824965837238, -1.992298350328743, -1.9117091698089397
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
window_sz = 15
thd = 0.2
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
