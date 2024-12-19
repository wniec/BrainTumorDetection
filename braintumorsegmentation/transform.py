import os.path
import shutil
import os
import nipype.interfaces.fsl as fsl


def bet_transform(patient_name):
    src_path = os.path.join("registered", patient_name)
    dst_path = os.path.join("no_skull", patient_name)
    os.mkdir(dst_path)
    bet1 = fsl.BET(
        in_file=os.path.join(src_path, "T1.nii.gz"),
        out_file=os.path.join(dst_path, "T1.nii.gz"),
    )
    bet1.inputs.output_type = "NIFTI_GZ"
    bet1.run()
    bet2 = fsl.BET(
        in_file=os.path.join(src_path, "T2.nii.gz"),
        out_file=os.path.join(dst_path, "T2.nii.gz"),
    )
    bet2.inputs.output_type = "NIFTI_GZ"
    bet2.run()
    if os.path.exists(os.path.join(src_path, "FLAIR.nii.gz")):
        bet3 = fsl.BET(
            in_file=os.path.join(src_path, "FLAIR.nii.gz"),
            out_file=os.path.join(dst_path, "FLAIR.nii.gz"),
        )
        bet3.inputs.output_type = "NIFTI_GZ"
        bet3.run()
    shutil.rmtree(src_path)


def register(patient: str):
    input_path = os.path.join("input", patient)
    output_path = os.path.join("registered", patient)
    os.mkdir(output_path)

    t1 = os.path.join(input_path, "T1.nii.gz")
    t2 = os.path.join(input_path, "T2.nii.gz")
    flair = os.path.join(input_path, "FLAIR.nii.gz")

    flt = fsl.FLIRT(bins=640, cost_func="mutualinfo")
    flt.inputs.in_file = t1
    flt.inputs.reference = t2
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.out_file = os.path.join(output_path, "T1.nii.gz")
    flt.inputs.dof = 12
    flt.inputs.out_matrix_file = "subject_to_template.mat"
    flt.run()

    if os.path.exists(flair):
        flt = fsl.FLIRT(bins=640, cost_func="mutualinfo")
        flt.inputs.in_file = flair
        flt.inputs.reference = t2
        flt.inputs.output_type = "NIFTI_GZ"
        flt.inputs.out_file = os.path.join(output_path, "FLAIR.nii.gz")
        flt.inputs.dof = 12
        flt.inputs.out_matrix_file = "subject_to_template.mat"
        flt.run()

    shutil.move(
        src=os.path.join(input_path, "T2.nii.gz"),
        dst=os.path.join(output_path, "T2.nii.gz"),
    )
    shutil.rmtree(input_path)
    os.remove("subject_to_template.mat")
