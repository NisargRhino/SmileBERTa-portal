import os
import shutil
import tempfile
from rdkit import Chem
from dockstring import load_target

def canonicalize_smiles(smiles):
    """Convert SMILES to its canonical form."""
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
    except Exception as e:
        print(f"Error canonicalizing SMILES {smiles}: {e}")
        return None

def prepare_docking_directory(docking_dir, mol_name, mol2_path, center_coords, box_sizes):
    """Prepare the docking directory and configuration files."""
    os.makedirs(docking_dir, exist_ok=True)
    
    # Convert mol2 to pdbqt
    mol2_pdbqt_path = os.path.join(docking_dir, mol_name + '_target.pdbqt')
    os.system(f"obabel -imol2 {mol2_path} -opdbqt -O {mol2_pdbqt_path} -xr")

    # Create the conf file
    conf_file_path = os.path.join(docking_dir, mol_name + '_conf.txt')
    with open(conf_file_path, 'w') as f:
        f.write(f"""center_x = {center_coords[0]}
center_y = {center_coords[1]}
center_z = {center_coords[2]}

size_x = {box_sizes[0]}
size_y = {box_sizes[1]}
size_z = {box_sizes[2]}""")
    return mol2_pdbqt_path, conf_file_path

def dock_ligand(smiles, docking_dir, mol2_path, center_coords, box_sizes):
    """Dock a single ligand and return the docking score."""
    mol_name = 'mtor'
    mol2_pdbqt_path, conf_file_path = prepare_docking_directory(docking_dir, mol_name, mol2_path, center_coords, box_sizes)

    target = load_target(mol_name, targets_dir=docking_dir)
    try:
        score, __ = target.dock(smiles)
        return score
    except Exception as e:
        print(f"Error docking SMILES {smiles}: {e}")
        return str(e)

def run_docking(smiles):
    docking_dir = tempfile.mkdtemp()
    mol2_path = "mtor.mol2"
    center_coords = [68.0658, -5.1678, -54.97]
    box_sizes = [98.194, 95.5592, 116.24]

    try:
        score = dock_ligand(smiles, docking_dir, mol2_path, center_coords, box_sizes)
        os.remove('input_structure.pdb')
        os.remove('fragment_structure.pdb')
        return score
    finally:
        shutil.rmtree(docking_dir)


if __name__ == "__main__":
    # Define the SMILES string to test
    smiles = "C1C(C(C(C(C1N)OC2C(C(C(C(O2)CN)O)O)O)O)OC3C(C(C(C(O3)CO)O)N)O)N"  # Example SMILES string for kanamycin
    score = run_docking(smiles)
    print(f"Best docking score: {score}")
