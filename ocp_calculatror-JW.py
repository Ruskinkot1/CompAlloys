#%%
import os
import random
from ase.build import bulk, molecule, add_adsorbate, surface
from ase import Atoms
from ase.io.vasp import write_vasp
from ase.optimize import LBFGS
from ase.io import Trajectory
from ase.constraints import FixAtoms
from fairchem.core.trainers.ocp_trainer import OCPTrainer
import fairchem.core.models.equiformer_v2.trainers.forces_trainer as forces_trainer 
from fairchem.core import OCPCalculator  
import pandas as pd
from tqdm import tqdm


# Configuration
elements = ['Co', 'Ni', 'Pd', 'Rh', 'Ru']  # Example elements
slab_size = (6, 6, 3)
num_samples = 2000
main_output_dir = "outputs"
os.makedirs(main_output_dir, exist_ok=True)

# Input/output directories for raw and relaxed structures
input_dir = os.path.join("data", "input")
output_dir = os.path.join("data", "output")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Adsorbate definitions
adsorbate_names = ['CH4', 'CH3', 'CO', 'CO2', 'CH', 'C', 'HCO', 'CH3O', 'H', 'H2', 'H2O', 'OH']
adsorbates = {name: molecule(name) for name in adsorbate_names}
adsorbates.update({
    'CH2': Atoms('CH2', positions=[[0,0,0], [0,0,1.09], [0,0,-1.09]]),
    'O': Atoms('O', positions=[[0,0,0]]),
    'HCOO': Atoms('HCOO', positions=[[0,0,0], [0,0,1.23], [0,0,-1.23], [0,0,1.0]]),
    'COOH': Atoms('COOH', positions=[[0,0,0], [0,0,1.23], [0,0,-1.23], [0,0.97,-1.23]]),
    'CH2O': Atoms('CH2O', positions=[[0,0,0], [0,0,1.21], [0,0.94,-0.54], [0,-0.94,-0.54]])
})

# Adsorption sites (relative positions on the surface)
positions = [(6,6), (6,7), (5,7), (7,5), (5,6), (6,5), (5,5), (7,7)]

# Placeholder for results
results = []

# Define your calculator
calc = OCPCalculator(
    model_name="EquiformerV2-83M-S2EF-OC20-2M",
    local_cache="pretrained_models",
    cpu=False
)

# Generate slabs — you need to define this function
def generate_hea_slabs(elements, slab_size, num_samples):
    num_atoms = slab_size[0] * slab_size[1] * slab_size[2]
    slabs = set()

    while len(slabs) < num_samples:
        composition = [random.choice(elements) for _ in range(num_atoms)]
        composition_tuple = tuple(sorted(composition))

        if composition_tuple not in slabs:
            slabs.add(composition_tuple)

            base_structure = bulk(elements[0], 'fcc', a=3.9)
            slab_structure = surface(base_structure, (1, 1, 1), layers=slab_size[2]).repeat((slab_size[0], slab_size[1], 1))
            for atom, element in zip(slab_structure, composition):
                atom.symbol = element
            yield slab_structure
# Main loop
i=0
for slab in tqdm(generate_hea_slabs(elements, slab_size, num_samples)):
    i=i+1
    combo_name = "".join(str(slab.get_chemical_formula())) + "-FCC"
    combo_dir = os.path.join(main_output_dir, combo_name)
    os.makedirs(combo_dir, exist_ok=True)

    slab.set_pbc([True, True, False])

    for ads_name, ads in adsorbates.items():
        
        for pos in positions:
            # File paths
            traj_path = os.path.join(combo_dir, f"slab_{i+1}_{ads_name}_{pos[0]}_{pos[1]}.traj")
            poscar_filename = os.path.join(output_dir, f"relaxed_slab_{i+1}_{ads_name}_{pos[0]}_{pos[1]}.vasp")
            input_filename = os.path.join(input_dir, f"input_slab_{i+1}_{ads_name}_{pos[0]}_{pos[1]}.vasp")

            # Copy and modify structure
            slab_ads = slab.copy()
            add_adsorbate(slab_ads, ads, height=1.8, position=pos)
            slab_ads.center(axis=2, vacuum=10)
            constraint = FixAtoms(indices=range(len(slab)))
            slab_ads.set_constraint(constraint)
            slab_ads.calc = calc

            # Save structure before relaxation
            write_vasp(input_filename, slab_ads, direct=True, sort=True, vasp5=True)

            # Relax the system
            dyn = LBFGS(slab_ads, trajectory=traj_path)
            dyn.run(fmax=0.05, steps=200)

            # Get energy and save relaxed structure
            energy = slab_ads.get_potential_energy()
            write_vasp(poscar_filename, slab_ads, direct=True, sort=True, vasp5=True)

            # Record result
            results.append({
                "slab_index": i+1,
                "element_combo": "-".join(slab.get_chemical_formula()),
                "adsorbate": ads_name,
                "position": pos,
                "energy": energy,
                "traj_file": traj_path,
                "input_file": input_filename,
                "output_file": poscar_filename
            })

    # Periodic saving
    if (i + 1) % 50 == 0:
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(main_output_dir, "energy_results.csv"), index=False)
        print(f"Processed {i+1} slabs. Current combo: {combo_name}")

# Final save
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(main_output_dir, "energy_results.csv"), index=False)
print("All tasks completed.")

#%%
slab.get_chemical_formula()
#%%
from fairchem.core.models.model_registry import available_pretrained_models

#%%
available_pretrained_models
#%%
from fairchem.core.models.model_registry import model_name_to_local_file
checkpoint_path = model_name_to_local_file('EquiformerV2-153M-S2EF-OC20-All+MD', local_cache='/tmp/fairchem_checkpoints/')
checkpoint_path
#%%
