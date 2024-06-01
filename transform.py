import os.path
import shutil
import os
import nipype.interfaces.fsl as fsl


def bet_transform(patient_name):
    src_path = os.path.join("registered", patient_name)
    dst_path = os.path.join("skull_segmented", patient_name)
    os.mkdir(dst_path)
    bet = fsl.BET(
        in_file=os.path.join(src_path, "T1.nii.gz"),
        out_file=os.path.join(dst_path, "T1.nii.gz"),
    )
    bet.inputs.output_type = "NIFTI_GZ"
    bet.run()
    bet = fsl.BET(
        in_file=os.path.join(src_path, "T2.nii.gz"),
        out_file=os.path.join(dst_path, "T2.nii.gz"),
    )
    bet.inputs.output_type = "NIFTI_GZ"
    bet.run()


def register(patient: str):
    input_path = os.path.join("input", patient)
    output_path = os.path.join("registered", patient)
    os.mkdir(output_path)
    t1 = os.path.join(input_path, "T1.nii.gz")
    t2 = os.path.join(input_path, "T2.nii.gz")
    flt = fsl.FLIRT(bins=640, cost_func="mutualinfo")
    flt.inputs.in_file = t1
    flt.inputs.reference = t2
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.out_file = os.path.join(output_path, "T1.nii.gz")
    flt.inputs.dof = 12
    flt.inputs.out_matrix_file = "subject_to_template.mat"
    flt.run()
    shutil.move(
        src=os.path.join(input_path, "T2.nii.gz"),
        dst=os.path.join(output_path, "T2.nii.gz"),
    )
    os.remove(os.path.join(input_path, "T1.nii.gz"))
    os.remove("subject_to_template.mat")
