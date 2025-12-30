import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#TUDO: solve this import
# from environment.strategies.rgc_mpc.base_controller import LogCompletionDetector

WINDOW_SZ = 30
THD = 0.05
task1_extended = [
    -2.4783817986652306, -0.13469220886110728, 0.19823026624738205, 0.25448470596451495, 0.26865090099726985,
    0.2716312826475701, 0.26890202399610375, 0.26322978637672206, 0.25625163288514996, 0.24892238197517297,
    0.24165500234970194, 0.23479393847545282, 0.22818548863144036, 0.22200111582720866, 0.21598379933807668,
    0.21032890750825972, 0.2049807258440616, 0.19935774138829715, 0.19369483938873655, 0.18854749990211153,
    0.18330791930721432, 0.17826893041804784, 0.17342872009435945, 0.16880652445625235, 0.16421798393120743,
    0.1594076408846534, 0.15505365186341918, 0.1511514178287938, 0.14695938949136284, 0.14368402359893354,
    0.13989195452115186, 0.1366874737793881, 0.13311527156049746, 0.12978335332703592, 0.12621639133878065,
    0.12382538257549358, 0.12068660272143678, 0.11799716501418853, 0.11492443960185647, 0.11236713703532095,
    0.10971946577669708, 0.10736443180056124, 0.10518477364646096, 0.103044972670624, 0.10094514953859988,
    0.0988784925828718, 0.09676259904966554, 0.09449983074665726, 0.0925203308799787, 0.09060300657841484,
    0.08871853433554845, 0.08693471463450556, 0.08514632453232715, 0.08339204013807289, 0.08164008563981955,
    0.07999292487722934, 0.07824822346465679, 0.07676913250905928, 0.07509414928773794, 0.07364214987569359,
    0.07205691928732444, 0.0705414272425943, 0.06905871918951191, 0.06750417164674796, 0.06604956909250069,
    0.06451104734620246, 0.06310266701402309, 0.061674718857109415, 0.0602409280511296, 0.05878773699190776,
    0.05736455443606814, 0.0559467540093137, 0.054560940240470805, 0.053143744221564065, 0.051674602258989434,
    0.05020575982170962, 0.04875494972789811, 0.04729392022323746, 0.04582040569747075, 0.04435773972428247,
    0.04291566075578114, 0.041458276795157574, 0.040047155832928005, 0.03866611780572672, 0.0373102149692298,
    0.03597518640724982, 0.03467252588249544, 0.033374430661238276, 0.03211978054334924, 0.03087992537801306,
    0.02966562211686739, 0.028481346756963084, 0.02729829082819297, 0.026150980934775914, 0.025033429498155995,
    0.023937286170224236, 0.022858306604181266, 0.021793917972806633, 0.020766809508106703, 0.019805907904182983,
    0.018805413762582032, 0.017832741729281967, 0.016960700696138142, 0.016029334975625482, 0.015142230654051196,
    0.014313995583502489, 0.013544907994700092, 0.012755372838822873, 0.01200852317666796, 0.011318893823239,
    0.01064067378646822, 0.009996744759968182, 0.009384193962443908, 0.008776793648268331, 0.00820472651075199,
    0.007682076950504106, 0.00718410527673988, 0.006684400681668368, 0.00620510164405184, 0.005761172712975561,
    0.0053186960102277045, 0.004900277131357458, 0.004484323481887066, 0.004087200151166141, 0.0037120656698136183,
    0.0033550435061958195, 0.0030150481457853076, 0.002712922053368366, 0.0023940636234564758, 0.002098192860118428,
    0.0018115531396250314, 0.0015355000331394532, 0.0012842819295432585, 0.0010271581023709078, 0.0007904344676061158,
    0.0005576329694336447, 0.0003404754923644953, 0.00013663966911183658, -5.007722318520253e-05,
    -0.00022401789090641032, -0.0003953575247581052, -0.0005698586422522824, -0.0007488285360122419,
    -0.000908373853714312, -0.001051561488749275, -0.0011765095380729517, -0.0012898749677144852,
    -0.0014143291341526896, -0.0015472384456181021, -0.0016479616294603444, -0.001731247353734092,
    -0.0018049818009194109, -0.0018853954665347078, -0.001955998029467591, -0.002023227447991573,
    -0.0020826359562237024, -0.0021198457777475126, -0.0021630451927956763, -0.002193180514953413,
    -0.002223098844735719, -0.0022533284992154816, -0.002282574052259583, -0.002304089394889733, -0.0023199104152220227,
    -0.0023266720681624415, -0.0023248671280036424, -0.002314975008342945, -0.002305745032143396,
    -0.0022870210105603267, -0.0022623928101654636, -0.00223391857634838, -0.0022043597229206304, -0.002175731669815136,
    -0.002149775629018467, -0.0021211164969287973
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
