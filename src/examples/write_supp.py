# Example usage
data = SupplementaryData(
    name="MyMolecule",
    point_group="C2v",
    energy=-76.12345,
    coordinates=coords,
    atomic_numbers=atomic_nums,
    symmetry_labels=symm_labels,
    frequencies=freqs,
    ir_intensities=ir_ints,
    zero_point_lengths=zpt_lens,
    normal_modes=modes
)

writer = SupplementaryWriter()
writer.write_supplementary_data(data, Path("output_dir"))