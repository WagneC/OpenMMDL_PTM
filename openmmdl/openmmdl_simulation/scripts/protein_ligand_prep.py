import os
import mdtraj as md
import numpy as np
import parmed as pmd
import simtk.openmm.app as app
from rdkit import Chem
from openff.toolkit.topology import Molecule
from simtk.openmm.app import PDBFile
from simtk.openmm import unit
from simtk.openmm import Vec3
from openff.interchange.components._packmol import (RHOMBIC_DODECAHEDRON, UNIT_CUBE,solvate_topology)
from openff.units import Quantity 
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler
from openff.toolkit import Molecule
from contextlib import contextmanager
from rdkit.Chem import rdCIPLabeler




def prepare_ligand(ligand_file, sanitization=False, minimize_molecule=True):
    """Reads an SDF File into RDKit, adds hydrogens to the structure, minimizes it if selected, and creates an openforcefield Molecule object. Inspired by @teachopencadd T019.

    Args:
        ligand_file (str): User input of SDF or MOL File.
        minimize_molecule (bool): Minimization of ligand.

    Returns:
        rdkitmolh (rdkit.Chem.rdchem.Mol): The prepared and converted ligand.
    """
    # Reading of SDF File, converting to rdkit.
    file_name = ligand_file.lower()
    if file_name.endswith(".sdf"):
        if sanitization:
            rdkit_mol = Chem.SDMolSupplier(ligand_file, sanitize=True)
        else:
            rdkit_mol = Chem.SDMolSupplier(ligand_file, sanitize=False)
        for mol in rdkit_mol:
            rdkit_mol = mol
    elif file_name.endswith(".mol") and not file_name.endswith(".mol2"):
        if sanitization:
            rdkit_mol = Chem.rdmolfiles.MolFromMolFile(ligand_file, sanitize=True)
        else:
            rdkit_mol = Chem.rdmolfiles.MolFromMolFile(ligand_file, sanitize=False)
    elif file_name.endswith(".mol2"):
        if sanitization:
            rdkit_mol = Chem.rdmolfiles.MolFromMol2File(ligand_file, sanitize=True)
        else:
            rdkit_mol = Chem.rdmolfiles.MolFromMol2File(ligand_file, sanitize=False)
    # Adding of hydrogens and assigning chiral tags from the structure.
    print("Adding hydrogens")
    rdkitmolh = Chem.AddHs(rdkit_mol, addCoords=True)
    Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)

    # Minimizes the molecule with the MMFF94s Forcefield if selected.
    if minimize_molecule:
        Chem.rdForceFieldHelpers.MMFFOptimizeMolecule(mol=rdkitmolh, mmffVariant="MMFF94s", maxIters=2000)

    # Converting of the ligand from rdkit to an opeenforcefield Molecule object.
    Molecule(rdkitmolh)

    return rdkitmolh


def rdkit_to_openmm(rdkit_mol, name):
    """Convert an RDKit molecule to an OpenMM molecule. Inspired by @teachopencadd T019, @hannahbrucemcdonald and @glass-w.

    Args:
        rdkit_mol (rdkit.Chem.rdchem.Mol): RDKit molecule to convert.
        name (str): Molecule name.

    Returns:
        omm_molecule (simtk.openmm.app.Modeller): OpenMM modeller object holding the molecule of interest.
    """
    # convert RDKit to OpenFF
    off_mol = Molecule.from_rdkit(rdkit_mol)

    # add name for molecule
    off_mol.name = name

    # add names for atoms
    element_counter_dict = {}
    for off_atom, rdkit_atom in zip(off_mol.atoms, rdkit_mol.GetAtoms()):
        element = rdkit_atom.GetSymbol()
        if element in element_counter_dict.keys():
            element_counter_dict[element] += 1
        else:
            element_counter_dict[element] = 1
        off_atom.name = element + str(element_counter_dict[element])

    # convert from OpenFF to OpenMM
    off_mol_topology = off_mol.to_topology()
    mol_topology = off_mol_topology.to_openmm()

    for chain in mol_topology.chains():
        for residue in chain.residues():
            residue.name = name

    new_mol_positions = []

    # convert units from Ångström to Nanometers
    for mol_position in off_mol.conformers[0]:
        new_mol_positions.append(mol_position.magnitude / 10.0)

    # combine topology and positions in modeller object
    omm_mol = app.Modeller(mol_topology, new_mol_positions * unit.nanometers)

    return omm_mol


def merge_protein_and_ligand(protein, ligand):
    """Merge two OpenMM objects. Inspired by @teachopencadd T019.

    Args:
        protein (pdbfixer.pdbfixer.PDBFixer): Protein to merge.
        ligand (simtk.openmm.app.Modeller): Ligand to merge.

    Returns:
        complex_topology (simtk.openmm.app.topology.Topology): The merged topology.
        complex_positions (simtk.unit.quantity.Quantity): The merged positions.
    """
    # combine topologies
    md_protein_topology = md.Topology.from_openmm(protein.topology)  # using mdtraj for protein top
    md_ligand_topology = md.Topology.from_openmm(ligand.topology)  # using mdtraj for ligand top
    md_complex_topology = md_protein_topology.join(md_ligand_topology)  # add them together

    complex_topology = md_complex_topology.to_openmm()

    # combine positions
    total_atoms = len(protein.positions) + len(ligand.positions)

    # create an array for storing all atom positions as tupels containing a value and a unit
    # called OpenMM Quantities
    complex_positions = unit.Quantity(np.zeros([total_atoms, 3]), unit=unit.nanometers)
    complex_positions[: len(protein.positions)] = protein.positions  # add protein positions
    complex_positions[len(protein.positions) :] = ligand.positions  # add ligand positions

    return complex_topology, complex_positions


def write_ligand_with_partial_charges(topology, system, positions, ligand_name=None, ligand_names=None, ligand_files=None, output_file=None):
    """Write one or more ligands with assigned partial charges to MOL2 files."""
    if ligand_names is None:
        ligand_names = [ligand_name] if ligand_name else []

    if not ligand_names:
        print("No ligand_name set; skipping MOL2 export (Amber uploaded prmtop needs resname input).")
        return None

    written_files = []

    try:
        struct = pmd.openmm.load_topology(topology, system, positions)
        fallback_residues = [
            res for res in struct.residues
            if res.name not in {"HOH", "WAT", "NA", "CL", "K", "CA", "MG"}
        ][-len(ligand_names):]
        for current_ligand_name in ligand_names:
            lig = struct[f":{current_ligand_name}"]
            if len(lig.atoms) == 0:
                lig = None
                if fallback_residues:
                    residue = fallback_residues.pop(0)
                    lig = struct[f":{residue.idx + 1}"]
                if lig is None or len(lig.atoms) == 0:
                    raise ValueError(f"Could not locate ligand residue '{current_ligand_name}' in topology.")

            if output_file and len(ligand_names) == 1:
                current_output_file = output_file
            elif ligand_files and len(written_files) < len(ligand_files):
                stem = os.path.splitext(os.path.basename(ligand_files[len(written_files)]))[0]
                current_output_file = f"{stem}_pc.mol2"
            else:
                current_output_file = f"{current_ligand_name}_pc.mol2"
            lig.save(current_output_file, overwrite=True)
            print(f"Wrote ligand with partial charges '{current_output_file}'.")
            written_files.append(current_output_file)
        if len(written_files) == 1:
            return written_files[0]
        return written_files
    except Exception as e:
        print(f"Skipping write out of partial charge molecule due to error: {e}")
        return None



def water_padding_solvent_builder(
    model_water,
    forcefield,
    water_padding_distance,
    protein_pdb,
    modeller,
    water_positive_ion,
    water_negative_ion,
    water_ionicstrength,
    protein_name,
):
    """Build a Solvent Box with padding values.

    Args:
        model_water (str): Selected water model.
        forcefield (openmm.app.forcefield.ForceField): Selected Forcefield.
        water_padding_distance (float): Solvent padding distance.
        protein_pdb (pdbfixer.pdbfixer.PDBFixer): Protein as a pdbfixer object.
        modeller (openmm.app.modeller.Modeller): Complex as a modeller object.
        water_positive_ion (str): Positive ion.
        water_negative_ion (str): Negative ion.
        water_ionicstrength (float): Ionic strength.
        protein_name (str): Protein file name.

    Returns:
        modeller (openmm.app.modeller.Modeller): The complex with solvent.
    """
    # Writing out the protein without solvent
    with open(f"prepared_no_solvent_{protein_name}", "w") as outfile:
        PDBFile.writeFile(protein_pdb.topology, protein_pdb.positions, outfile)

    # Adds solvent to the selected protein
    if model_water == "charmm" or model_water == "tip3pfb" or model_water == "tip3":
        modeller.addSolvent(
            forcefield,
            padding=water_padding_distance * unit.nanometers,
            positiveIon=water_positive_ion,
            negativeIon=water_negative_ion,
            ionicStrength=water_ionicstrength * unit.molar,
        )
    elif model_water == "charmm_tip4pew":
        protein_pdb.addSolvent(
            padding=water_padding_distance * unit.nanometers,
            positiveIon=water_positive_ion,
            negativeIon=water_negative_ion,
            ionicStrength=water_ionicstrength * unit.molar,
        )
    else:
        if model_water in ("tip4pfb", "opc"):
            model_water = "tip4pew"
        elif model_water == "opc3":
            model_water = "tip3p"
        modeller.addSolvent(
            forcefield,
            model=model_water,
            padding=water_padding_distance * unit.nanometers,
            positiveIon=water_positive_ion,
            negativeIon=water_negative_ion,
            ionicStrength=water_ionicstrength * unit.molar,
        )

    # Writing out the protein with padding solvent
    with open(f"solvent_padding_{protein_name}", "w") as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)
    print("Protein with buffer solvent prepared")

    return modeller


def water_absolute_solvent_builder(
    model_water,
    forcefield,
    water_box_x,
    water_box_y,
    water_box_z,
    protein_pdb,
    modeller,
    water_positive_ion,
    water_negative_ion,
    water_ionicstrength,
    protein_name,
):
    """Build a Solvent Box with absolute values.

    Args:
        model_water (str): Selected water model.
        forcefield (openmm.app.forcefield.ForceField): Selected Forcefield.
        water_box_x (float): Vector x of the solvent box.
        water_box_y (float): Vector y of the solvent box.
        water_box_z (float): Vector z of the solvent box.
        protein_pdb (pdbfixer.pdbfixer.PDBFixer): Protein as a pdbfixer object.
        modeller (openmm.app.modeller.Modeller): Complex as a modeller object.
        water_positive_ion (str): Positive ion.
        water_negative_ion (str): Negative ion.
        water_ionicstrength (float): Ionic strength.
        protein_name (str): Protein file name.

    Returns:
        modeller (openmm.app.modeller.Modeller): The complex with solvent.
    """
    # Writing out the protein without solvent
    with open(f"prepared_no_solvent_{protein_name}", "w") as outfile:
        PDBFile.writeFile(protein_pdb.topology, protein_pdb.positions, outfile)

    # Adds solvent to the selected protein
    if model_water == "charmm" or model_water == "tip3pfb" or model_water == "tip3":
        modeller.addSolvent(
            forcefield,
            boxSize=Vec3(water_box_x, water_box_y, water_box_z) * unit.nanometers,
            positiveIon=water_positive_ion,
            negativeIon=water_negative_ion,
            ionicStrength=water_ionicstrength * unit.molar,
        )
    elif model_water == "charmm_tip4pew":
        protein_pdb.addSolvent(
            boxSize=Vec3(water_box_x, water_box_y, water_box_z) * unit.nanometers,
            positiveIon=water_positive_ion,
            negativeIon=water_negative_ion,
            ionicStrength=water_ionicstrength * unit.molar,
        )
    else:
        if model_water in ("tip4pfb", "opc"):
            model_water = "tip4pew"
        elif model_water == "opc3":
            model_water = "tip3p"
        modeller.addSolvent(
            forcefield,
            model=model_water,
            boxSize=Vec3(water_box_x, water_box_y, water_box_z) * unit.nanometers,
            positiveIon=water_positive_ion,
            negativeIon=water_negative_ion,
            ionicStrength=water_ionicstrength * unit.molar,
        )

    # Writing out the protein with absolute solvent
    with open(f"solvent_absolute_{protein_name}", "w") as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)
    print("Protein with absolute solvent prepared")

    return modeller


def membrane_builder(
    ff,
    model_water,
    forcefield,
    transitional_forcefield,
    protein_pdb,
    modeller,
    membrane_lipid_type,
    membrane_padding,
    membrane_positive_ion,
    membrane_negative_ion,
    membrane_ionicstrength,
    protein_name,
):
    """Build a membrane with minimum padding.

    Args:
        ff (str): Selected Forcefield as a string.
        model_water (str): Selected water model.
        forcefield (openmm.app.forcefield.ForceField): Selected Forcefield.
        transitional_forcefield (openmm.app.forcefield.ForceField): Transitional forcefield for specific water models.
        protein_pdb (pdbfixer.pdbfixer.PDBFixer): Protein as a pdbfixer object.
        membrane_lipid_type (str): Lipid type.
        membrane_padding (float): Minimum membrane padding distance.
        membrane_positive_ion (str): Positive ion.
        membrane_negative_ion (str): Negative ion.
        membrane_ionicstrength (float): Ionic strength.
        protein_name (str): Protein file name.

    Returns:
        modeller (openmm.app.modeller.Modeller): The complex with solvent.
    """
    # Writing out the protein without solvent
    with open(f"prepared_no_solvent_{protein_name}", "w") as outfile:
        PDBFile.writeFile(protein_pdb.topology, protein_pdb.positions, outfile)

    # Adds a membrane to the selected protein
    # The Water Models TIP4P and TIP5P require an transitional forcefield
    if ff == "CHARMM36":
        protein_pdb.addMembrane(
            lipidType=membrane_lipid_type,
            minimumPadding=membrane_padding * unit.nanometer,
            positiveIon=membrane_positive_ion,
            negativeIon=membrane_negative_ion,
            ionicStrength=membrane_ionicstrength * unit.molar,
        )
        modeller = app.Modeller(protein_pdb.topology, protein_pdb.positions)
    else:
        if model_water == "charmm":
            modeller.addMembrane(
                forcefield,
                lipidType=membrane_lipid_type,
                minimumPadding=membrane_padding * unit.nanometer,
                positiveIon=membrane_positive_ion,
                negativeIon=membrane_negative_ion,
                ionicStrength=membrane_ionicstrength * unit.molar,
            )
        else:
            virtual_site_membrane_waters = {
                "tip4pew",
                "tip4pfb",
                "tip5p",
                "opc",
                "charmm_tip4pew",
            }

            convertible_waters = {
                "tip4pew",
            }

            extra_particle_waters = {
                "tip4pfb",
                "opc",
                "tip5p",
                "charmm_tip4pew",
            }

            if model_water in virtual_site_membrane_waters:
                modeller.addMembrane(
                    transitional_forcefield,
                    lipidType=membrane_lipid_type,
                    minimumPadding=membrane_padding * unit.nanometer,
                    positiveIon=membrane_positive_ion,
                    negativeIon=membrane_negative_ion,
                    ionicStrength=membrane_ionicstrength * unit.molar,
                )

                if model_water in convertible_waters:
                    modeller.convertWater(model_water)
                elif model_water in extra_particle_waters:
                    modeller.addExtraParticles(forcefield)

            else:
                modeller.addMembrane(
                    forcefield,
                    lipidType=membrane_lipid_type,
                    minimumPadding=membrane_padding * unit.nanometer,
                    positiveIon=membrane_positive_ion,
                    negativeIon=membrane_negative_ion,
                    ionicStrength=membrane_ionicstrength * unit.molar,
                )

    with open(f"membrane_{protein_name}", "w") as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

    print(f"Protein with Membrane {membrane_lipid_type} prepared")

    return modeller


def water_conversion(model_water, modeller_pre_conversion, protein_name):
    """Convert the water model of an OpenMM object.

    Args:
        model_water (str): The name of the preferred converted Water model.
        modeller_pre_conversion (pdbfixer.pdbfixer.PDBFixer): The object that will be converted.
        protein_name (str): Name of the Protein pdb file.

    Returns:
        modeller (openmm.app.modeller.Modeller): The converted object.
    """
    # Writes out the PDB of the preconverted pdb
    with open(f"pre_converted_{protein_name}", "w") as outfile:
        PDBFile.writeFile(modeller_pre_conversion.topology, modeller_pre_conversion.positions, outfile)

    # Converts the water from TIP3P to the required Water Model
    modeller_pre_conversion.convertWater(model_water)
    modeller = modeller_pre_conversion

    # Writes out the pdb of the postconverted pdb
    with open(f"converted_{protein_name}", "w") as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

    return modeller

def solvate_topol_padding_openff(topology, water_padding_distance, water_boxShape, water_ionicstrength):
    BOX_SHAPES = {
        "cube": UNIT_CUBE,
        "dodecahedron": RHOMBIC_DODECAHEDRON,
    }
    topology.box_vectors = None

    return solvate_topology(
        topology,
        nacl_conc=Quantity(water_ionicstrength, "mol/L"),
        padding=Quantity(water_padding_distance, "nm"),
        box_shape=BOX_SHAPES[water_boxShape],
    )

def solvate_topol_absolute_openff(topology, water_box_x, water_box_y, water_box_z, water_ionicstrength):
    topology.box_vectors = Quantity(np.diag([water_box_x, water_box_y, water_box_z]), "nm")

    return solvate_topology(
    topology,
    nacl_conc=Quantity(water_ionicstrength, "mol/L"),
    padding=None,
    )

_CIS_TRANS = (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS)


def _cis_trans_to_e_z(rdmol):
    """Converts RDKit STEREOCIS/STEREOTRANS bond stereo to STEREOE/STEREOZ via CIP labels,
    since the OpenFF toolkit only understands E/Z.
    """
    if not any(b.GetStereo() in _CIS_TRANS for b in rdmol.GetBonds()):
        return rdmol
    rdmol = Chem.Mol(rdmol)  # copy, indices/props are kept
    # reaction products are unsanitized, CIP labeling needs ring info and valences
    rdmol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(rdmol)
    rdCIPLabeler.AssignCIPLabels(rdmol)
    for b in rdmol.GetBonds():
        if b.GetStereo() in _CIS_TRANS and b.HasProp("_CIPCode"):
            code = b.GetProp("_CIPCode")
            if code == "E":
                b.SetStereo(Chem.BondStereo.STEREOE)
            elif code == "Z":
                b.SetStereo(Chem.BondStereo.STEREOZ)
    return rdmol


@contextmanager
def _openff_accepts_cis_trans():
    """Temporarily patches Molecule.from_rdkit to accept cis/trans bond stereo."""
    # from_rdkit is usually inherited from FrozenMolecule, not defined on Molecule itself
    had_own = "from_rdkit" in Molecule.__dict__
    saved = Molecule.__dict__.get("from_rdkit")
    orig_func = Molecule.from_rdkit.__func__

    def _patched(cls, rdmol, *args, **kwargs):
        return orig_func(cls, _cis_trans_to_e_z(rdmol), *args, **kwargs)

    Molecule.from_rdkit = classmethod(_patched)
    try:
        yield
    finally:
        if had_own:
            Molecule.from_rdkit = saved
        else:
            del Molecule.from_rdkit


def react_ptm_residue(res_resdef, lig_resdef, res_smarts, lig_smarts, ptm_smarts):
    """Builds the PTM ResidueDefinition by reacting the residue with the ligand."""
    from openff.pablo import ResidueDefinition

    with _openff_accepts_cis_trans():
        return ResidueDefinition.react(
            reactants=[res_resdef, lig_resdef],
            reactant_smarts=[res_smarts, lig_smarts],
            product_smarts=[ptm_smarts],
        )[0][0]


def ptm_topology_from_pdb(protein, ptm_resdef):
    """Loads a PDB containing a PTM residue into an OpenFF Topology."""
    from openff.pablo import topology_from_pdb

    with _openff_accepts_cis_trans():
        topology = topology_from_pdb(protein, additional_definitions=[ptm_resdef])
    topology.box_vectors = None
    return topology


REFERENCE_OPENMM_FF = ("amber14-all.xml", "amber14/tip3p.xml")


def load_reference_forcefield(forcefield_selected, water_selected=None):
    """Load an OpenMM force field whose residue templates serve as a reference.

    OpenFF force fields (.offxml) do not contain residue templates, so in that case
    an OpenMM reference force field is used (for recognition only, not for parametrization).
    """
    files = [f for f in (forcefield_selected, water_selected) if f]
    if not files or any(f.endswith(".offxml") for f in files):
        print(f"OpenFF force field has no residue templates; using {REFERENCE_OPENMM_FF} as reference.")
        files = list(REFERENCE_OPENMM_FF)
    return app.ForceField(*files)


def _solvent_templates(reference_ff):
    """Read water and ion names from the templates of the force field."""
    water = None
    ions = {}
    for template in reference_ff._templates.values():
        atoms = [a for a in template.atoms if a.element is not None]  # ohne Virtual Sites
        symbols = sorted(a.element.symbol for a in atoms)
        if water is None and symbols == ["H", "H", "O"]:
            water = (template.name, [a.name for a in atoms])
        elif len(template.atoms) == 1 and atoms:
            ions.setdefault(atoms[0].element.symbol, (template.name, atoms[0].name))
    return water, ions


def rename_openff_solvent(topology, reference_ff):
    """Change names of water and ions to match the reference force field.
    """
    water, ions = _solvent_templates(reference_ff)
    if water is None:
        raise ValueError("No water template found in the reference force field (water XML missing?).")
    water_resname, water_atom_names = water
    water_o = [n for n in water_atom_names if n.upper().startswith("O")][0]
    water_h = [n for n in water_atom_names if n.upper().startswith("H")]

    for residue in topology.residues():
        atoms = list(residue.atoms())
        symbols = [a.element.symbol if a.element is not None else "" for a in atoms]
        if sorted(symbols) == ["H", "H", "O"]:
            residue.name = water_resname
            h_names = iter(water_h)
            for atom in atoms:
                atom.name = water_o if atom.element.symbol == "O" else next(h_names)
        elif len(atoms) == 1 and symbols[0] in ions:
            residue.name, atoms[0].name = ions[symbols[0]]
        else:
            counts = {}
            for atom, sym in zip(atoms, symbols):
                if not atom.name or not atom.name.strip():
                    counts[sym] = counts.get(sym, 0) + 1
                    atom.name = f"{sym or 'X'}{counts[sym]}"
    return topology


def get_ptm_residue_name(topology, ptm_name, reference_ff=None):
    """Check that the user-defined PTM residue is present in the topology.

    Returns:
        str: The PTM residue name, used as ligand name in post-processing/analysis.
    """
    ptm_name = ptm_name.strip().upper()
    ptm_residues = [res for res in topology.residues() if res.name == ptm_name]
    if not ptm_residues:
        found = sorted({res.name for res in topology.residues()})
        raise ValueError(
            f"PTM residue '{ptm_name}' not found in the topology. "
            f"Residue names present: {found}. Check the residue name in the input PDB."
        )
    print(
        f"Found {len(ptm_residues)} PTM residue(s) named '{ptm_name}' "
        f"(chain index, residue index: {[(r.chain.index, r.index) for r in ptm_residues]})."
    )

    if reference_ff is not None:
        others = sorted({r.name for r in reference_ff.getUnmatchedResidues(topology)} - {ptm_name})
        if others:
            print(
                "Warning: these residues also have no force field template and are NOT "
                f"treated as PTM/ligand in the analysis: {others}"
            )
    return ptm_name