"""Calculate anomalous peak heights at known anomalous scatterer positions.

Reads ANOM/PHANOM columns from a refined MTZ, builds the anomalous
difference map, and reports the peak height (in sigma) at each
anomalous atom site in the reference PDB.

Usage:
    python anomalous_peak_heights.py <mtz> <pdb> [S,I] <out.csv>
"""

import csv
import sys

import gemmi
import numpy as np


def get_anom_peak_heights(mtz_filename, pdb_filename, atom_list):
    mtz_file = gemmi.read_mtz_file(mtz_filename)
    st = gemmi.read_pdb(pdb_filename)

    real_grid = mtz_file.transform_f_phi_to_map("ANOM", "PHANOM", sample_rate=3.0)
    real_grid.normalize()

    sel = gemmi.Selection(f"{atom_list}")
    sel_model = sel.copy_model_selection(st[0])
    anom_atoms = list(sel_model.all())

    anom_res = []
    anom_peaks = []

    for cra in anom_atoms:
        eq_points = []
        ops = real_grid.spacegroup.operations()
        atom = cra.atom

        for op in ops:
            sg_mapped = op.apply_to_xyz(st.cell.fractionalize(atom.pos).tolist())
            tmp = sg_mapped - np.floor(np.array(sg_mapped))
            eq_points.append(gemmi.Fractional(*tmp))

        peak_value = []
        for pos in eq_points:
            a = round(pos.x * real_grid.nu)
            b = round(pos.y * real_grid.nv)
            c = round(pos.z * real_grid.nw)
            peak_value.append(real_grid.get_value(a, b, c))

        anom_height = round(float(np.average(peak_value)), 3)
        anom_res.append(f"{cra.residue.name} {cra.residue.seqid.num}")
        anom_peaks.append(anom_height)

    return anom_res, anom_peaks


def main():
    anom_res, anom_peaks = get_anom_peak_heights(
        mtz_filename=sys.argv[1],
        pdb_filename=sys.argv[2],
        atom_list=sys.argv[3],
    )
    with open(sys.argv[4], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(anom_res)
        writer.writerow(anom_peaks)


if __name__ == "__main__":
    main()
