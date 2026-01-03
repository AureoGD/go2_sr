import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#TUDO: solve this import
# from environment.strategies.rgc_mpc.base_controller import LogCompletionDetector

WINDOW_SZ = 35
THD = 0.1
task1_extended = [
    -67.23370833168461, -782.2248626700411, -902.146013457937, -817.219484545737, -736.809861957309, -681.1548844095965,
    -639.0129810490663, -604.7431484154652, -569.5451241508366, -539.256418862911, -509.17102426126786,
    -478.1575845729061, -444.35492865624366, -409.7145277466952, -392.326686228678, -350.647596839896,
    -329.830502394546, -305.55528711557633, -291.8140198508168, -276.4793041489232, -258.76086085171386,
    -243.09297927596066, -230.92501917761848, -235.57315113075637, -228.0458733034461, -220.1000342372356,
    -211.46020558894176, -212.7020763789498, -206.39033111180643, -198.45912759289916, -188.0589550866548,
    -181.10831665498105, -174.06843162092306, -167.83786995618453, -170.1192885826741, -165.75918444613419,
    -160.61386425578783, -154.75519286655134, -145.962277268794, -155.32995048280605, -155.7533797072175,
    -152.58098699350828, -148.1661319981874, -141.75292859344367, -141.17962811175428, -136.82337357475714,
    -135.70528430765864, -126.45590417894059, -124.68309637480102, -117.91125391565471, -109.86679840917897,
    -110.46298344399847, -101.74453924158317, -97.77963272162411, -98.25058785536275, -93.81686900049668,
    -92.90100319584383, -85.90557465769922, -83.970018547761, -83.63041773782398, -81.59405826008219,
    -76.33182777822222, -73.3832877362078, -71.82965623248141, -71.14019330492145, -69.3645145306776,
    -69.67028945582216, -65.18627594229098, -59.830459218018035, -61.365790131556146, -60.4182808875868,
    -55.100184790614406, -56.52316867897498, -56.8032051566118, -57.75499616827859, -55.310384004618726,
    -52.877027830243726, -54.09365295346234, -52.81481949244964, -52.592668916078885, -45.754604774763635,
    -50.336706425658036, -50.83731121752836, -50.731434386731095, -48.315645558553506, -47.617424124639264,
    -47.270998070292556, -47.29920817961094, -47.306815767476635, -46.570253552077936, -45.84637855181945,
    -44.352927348973445, -44.60039983795166, -42.32168757941668, -41.74847411939815, -41.258962309537345,
    -41.30194358671695, -42.35123249494323, -39.745322835505526, -40.9969266905368, -38.98621430017464,
    -37.52107949951353, -37.663828451107136, -38.12450554051415, -40.65925158304254, -35.88396484521473,
    -35.1219997455841, -37.29993153768389, -37.15736987516368, -37.4477380436966, -40.96868379430836,
    -38.02620921101509, -36.74961773188429, -38.93198241345531, -39.042835011654105, -39.01526582017957,
    -42.42017787681008, -41.69848433227173, -42.55894428095639, -44.72698878274654, -41.434956125249286,
    -43.12758274180502, -42.35428265393538, -44.11767373140689, -43.544404754087985, -44.12426698552196,
    -44.79906945038295, -45.26141497390368, -45.18386520487679, -42.910324324182916, -42.58933469289963,
    -42.77987214400027, -43.40427251158596, -41.361828328567, -41.68690908623666, -41.003140020508354,
    -40.054219003676614, -38.8098571194103, -38.25911442803405, -37.971303885162726, -36.90133448577469,
    -36.099101602026366, -35.317712025494686, -34.43299666320393, -33.27821854851267, -32.492970022924304,
    -31.60286500860936, -30.629048042858614, -29.744689886648683, -29.526122727365955, -28.882952923151358,
    -27.805489311352602, -27.631058119329854, -26.871048618233228, -26.60686074011143, -26.253898071702352,
    -26.102627330062017, -25.873435969073554, -24.91490460774233, -24.632040309008218, -24.376870076055337,
    -24.947432796194192, -24.947432796194192, -24.947432796194192, -21.246295636766977, -23.47050163709862,
    -23.80064266956289, -22.70421442896204, -24.24620879786314, -24.197977230431313, -24.09411378725467,
    -23.667163397058204, -21.70346913009025, -21.803487553968605, -21.63673493900791, -22.106736408760177,
    -21.86421348529546, -20.587082219713874, -21.484161903519738, -22.294249272923075, -21.21194057992856,
    -20.793256521056293, -19.970858683385924, -20.81632884656645, -21.002878347996326, -21.56737007138083,
    -21.56737007138083, -21.56737007138083, -21.56737007138083, -21.56737007138083, -21.56737007138083,
    -21.56737007138083, -21.56737007138083, -21.56737007138083, -21.56737007138083, -21.56737007138083,
    -21.56737007138083, -21.56737007138083, -21.56737007138083, -21.56737007138083
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
