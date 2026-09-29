import os
import napari
import pandas as pd
from os import chmod
from skimage.io import imsave
from skimage.filters import gaussian
from skimage.feature import blob_dog as dog, peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from scipy.ndimage import zoom, minimum_filter, median_filter, distance_transform_edt
from scipy.ndimage import center_of_mass
from utils import *

@magicgui(call_button='Seed Nuclei',
          nuc_scale_minmax = {'widget_type': 'RangeSlider', 'min': 1, 'max': 16, 'step': 1, 'readout': False, 'label': 'scale range'},
          nuc_det_thr={'widget_type': 'FloatSlider', 'min': 0, 'max': 2},
          nuc_merge_maxdst={'widget_type': 'IntSlider', 'min': 1, 'max': 250},
          memb_mindelta={'widget_type': 'IntSlider', 'min': 1, 'max': 100},
          load_labels={'widget_type': 'CheckBox', 'label': 'load labels (use centroids as seeds)'})
def seed_nuclei(vw: Viewer, nuc_scale_minmax = (4, 12), nuc_det_thr = 1, nuc_merge_maxdst=100, memb_mindelta=25, load_labels=False):

    if viewer_is_layer(vw, 'Nuclei') and viewer_is_layer(vw, 'Membrane'):

        zratio = load_image_tiff.zratio.value

        if load_labels:

            image_file = str(load_image_tiff.imagefile.value)[:-4]+'_nuc_lbl.tif'

            if os.path.isfile(image_file):
                nuclei_lbl = imread(image_file).astype(np.uint16)
                labels = np.unique(nuclei_lbl)
                coords_kept = center_of_mass(np.ones_like(nuclei_lbl), labels=nuclei_lbl, index=labels[labels != 0])
            else:
                print('!! Error: No "Nuclei" layer not found !!')

        else:

            #### XY downscaling factor for nuclei detection
            prescale = 0.5

            #### Fetch data from Napari objects
            nuclei = vw.layers['Nuclei'].data
            membrane = vw.layers['Membrane'].data

            #### Detect seeds (nucleus channel)
            # Multiscale DoG
            print('-------------------------------')
            print('Performing nucleus detection...')
            print('DoG detection')
            blobs = dog(zoom(nuclei, (zratio*prescale, prescale, prescale), order=1), min_sigma=nuc_scale_minmax[0]*prescale,
                        max_sigma=nuc_scale_minmax[1]*prescale, sigma_ratio=1.6, overlap=0.5, threshold=nuc_det_thr*1e-3, exclude_border=False)
            coords = [(int(blob[0]/(zratio*prescale)), int(blob[1]/prescale), int(blob[2]/prescale)) for blob in blobs]
            print(f"Found {len(coords)} candidate seeds")
            # Merge seeds without significant membrane signal in between
            print('Membrane raytracing_iter1')
            coords_kept = remove_seeds(membrane, coords, nuc_merge_maxdst*prescale, memb_mindelta, zratio)
            print('Membrane raytracing_iter2')
            coords_kept = remove_seeds(membrane, coords_kept, nuc_merge_maxdst*prescale, memb_mindelta, zratio)
            print('Membrane raytracing_iter3')
            coords_kept = remove_seeds(membrane, coords_kept, nuc_merge_maxdst*prescale, memb_mindelta, zratio)
            print(f"Kept {len(coords_kept)} seeds ({len(coords_kept)/len(coords):0.3f})")

            #### Add raw seeds to napari layers
            if viewer_is_layer(vw, "Seeds"):
                vw.layers["Seeds"].data = coords
            else:
                vw.add_points(coords, name=f"Seeds", size=15, face_color='black', blending="additive", scale=(zratio, 1, 1))
            vw.layers['Seeds'].visible=False

        #### Add kept seeds to napari layers
        if viewer_is_layer(vw, "Seeds_Kept"):
            vw.layers["Seeds_Kept"].data = coords_kept
        else:
            vw.add_points(coords_kept, name=f"Seeds_Kept", size=15, face_color='green', blending="additive", scale=(zratio, 1, 1))

    else:
        print('!! No "Nuclei" layer found !!')

@magicgui(call_button='Label Cells',
          cell_gaussrad={'widget_type': 'FloatSlider', 'min': 0, 'max': 1.5},
          cell_regrad={'widget_type': 'IntSlider', 'min': 1, 'max': 9},
          cell_minvol={'widget_type': 'IntSlider', 'min': 1, 'max': 5e3},
          cell_maxvol={'widget_type': 'IntSlider', 'min': 1, 'max': 2e5})
def segment_cells(vw: Viewer, cell_gaussrad=0.5, cell_regrad=5, cell_minvol=2e3, cell_maxvol=1e5):

    if viewer_is_layer(vw, 'Seeds_Kept') and viewer_is_layer(vw, 'Membrane'):

        #### Fetch data from Napari objects
        imagefile = load_image_tiff.imagefile.value
        zratio = load_image_tiff.zratio.value
        membrane = vw.layers['Membrane'].data
        coords_kept = [tuple(row.astype(int)) for row in vw.layers['Seeds_Kept'].data]

        #### Hide seed layers
        if viewer_is_layer(vw, 'Seeds'):
            vw.layers['Seeds'].visible = False
        vw.layers['Seeds_Kept'].visible = False

        #### Segment cells
        print('-------------------------------')
        print('Performing cell segmentation...')
        print('Filtering membrane signal')
        membrane_flt = gaussian(membrane.astype(float), sigma=(cell_gaussrad, cell_gaussrad, cell_gaussrad), preserve_range=True).astype('uint16')
        print('Imposing regional minima')
        seeds = seedcoords2mask(coords_kept, membrane.shape)
        membrane_imp = imposemin(membrane_flt, seeds>0)
        print('Watersheding')
        cell_lbl = watershed(membrane_imp, seeds, compactness=0)
        del membrane_imp
        print('Smoothing cells')
        cell_lbl = median_filter(cell_lbl, size=(1, cell_regrad, cell_regrad))
        print('Removing small/large cells')
        cell_lbl = remove_lbl_size(cell_lbl, cell_minvol, cell_maxvol)
        print('Removing cells touching borders')
        cell_lbl = remove_lbl_edge(cell_lbl)
        print('Filling cell holes')
        cell_lbl = fill_lbl_holes(cell_lbl)
        print('Relabeling cells')
        cell_lbl = relabel(cell_lbl)
        print('Measuring cells')
        properties = regionprops_table(cell_lbl, properties=['label', 'centroid', 'area'])

        #### Export results
        print('-------------------------------')
        print('Exporting results...')
        imsave(str(imagefile).replace('.tif', '_lbl.tif'), np.uint16(cell_lbl), check_contrast=False)
        chmod(str(imagefile).replace('.tif', '_lbl.tif'), 0o666)
        print(str(imagefile).replace('.tif', '_lbl.tif'))
        df = pd.DataFrame(properties)
        df.columns = ['Cell', 'CZ', 'CY', 'CX', 'Volume (vox)']
        df.to_csv(str(imagefile).replace('.tif', '_lbl.csv'), index=False)
        chmod(str(imagefile).replace('.tif', '_lbl.csv'), 0o666)
        print(str(imagefile).replace('.tif', '_lbl.csv'))
        print('-------------------------------')
        print(f'Number of segmented cells: {len(df)}')

        #### Add cell label layer
        np.random.seed(0)
        cell_lbl = cell_lbl * (cell_lbl == minimum_filter(cell_lbl, size=(1, 3, 3)))
        if viewer_is_layer(vw, "CellsLbl"):
            vw.layers["CellsLbl"].data = cell_lbl
        else:
            vw.add_labels(cell_lbl, name=f"CellsLbl", blending="additive", scale=(zratio, 1, 1))
        # Compute cell meshes from label mask (slow)
        #combined_verts, combined_faces, combined_values, custom_colormap = lbl2mesh(cell_lbl)
        #viewer.add_surface((combined_verts, combined_faces, combined_values), name='Combined Surface', colormap=custom_colormap, scale=(zratio, 1, 1))
        vw.layers.selection.active = viewer.layers['CellsLbl']

    else:
        print('!! Layer "Seeds_Kept" / "Membrane" not found !!')


@magicgui(call_button='Label Nuclei',
          nucleus_gaussrad={'widget_type': 'FloatSlider', 'min': 0, 'max': 3},
          nucleus_thr={'widget_type': 'IntSlider', 'min': 0, 'max': 9999},
          nucleus_regrad={'widget_type': 'IntSlider', 'min': 1, 'max': 9})
def segment_nuclei(vw: Viewer, nucleus_gaussrad=1.5, nucleus_thr=500, nucleus_regrad=5):

    if viewer_is_layer(vw, 'Nuclei') and viewer_is_layer(vw, 'CellsLbl'):

        #### Fetch data from Napari objects
        imagefile = load_image_tiff.imagefile.value
        zratio = load_image_tiff.zratio.value
        nuclei = vw.layers['Nuclei'].data
        cell_lbl = vw.layers['CellsLbl'].data

        print('-------------------------------')
        print('Performing nuclei segmentation...')
        print('Filtering and thresholding')
        nuclei_thr = gaussian(nuclei.astype(float), sigma=(nucleus_gaussrad/zratio, nucleus_gaussrad, nucleus_gaussrad),
                              preserve_range=True).astype('uint16') >= nucleus_thr
        print('Smoothing nuclei')
        nuclei_thr = median_filter(nuclei_thr, size=(1, nucleus_regrad, nucleus_regrad))
        print('Combining masks')
        nucleus_lbl = cell_lbl*nuclei_thr
        print('Filling nuclei holes')
        nucleus_lbl = fill_lbl_holes(nucleus_lbl)
        print('Measuring nuclei')
        properties = regionprops_table(nucleus_lbl, properties=['label', 'centroid', 'area'])

        #### Export results
        print('-------------------------------')
        print('Exporting results...')
        imsave(str(imagefile).replace('.tif', '_nuc_lbl.tif'), np.uint16(cell_lbl), check_contrast=False)
        chmod(str(imagefile).replace('.tif', '_nuc_lbl.tif'), 0o666)
        print(str(imagefile).replace('.tif', '_nuc_lbl.tif'))
        df = pd.DataFrame(properties)
        df.columns = ['Cell', 'CZ', 'CY', 'CX', 'Volume (vox)']
        df.to_csv(str(imagefile).replace('.tif', '_nuc_lbl.csv'), index=False)
        chmod(str(imagefile).replace('.tif', '_nuc_lbl.csv'), 0o666)
        print(str(imagefile).replace('.tif', '_nuc_lbl.csv'))
        print('-------------------------------')
        print(f'Number of segmented nuclei: {len(df)}')

        #### Add nucleus label layer
        np.random.seed(0)
        if viewer_is_layer(vw, "NucleiLbl"):
            vw.layers["NucleiLbl"].data = nucleus_lbl
        else:
            vw.add_labels(nucleus_lbl, name="NucleiLbl", blending="additive", scale=(zratio, 1, 1))
        vw.layers.selection.active=viewer.layers['NucleiLbl']

    else:
        print('!! Layer "Nuclei" / "CellsLbl" not found !!')


# Instantiate Napari viewer and add widgets
viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(load_image_tiff, area='right', name='Load Image')
dw1.setMinimumHeight(160);dw1.setMaximumHeight(160);dw1.setMinimumWidth(360)
dw2 = viewer.window.add_dock_widget(seed_nuclei, area='right', name='Seed Nuclei')
dw2.setMinimumHeight(220);dw2.setMaximumHeight(220)
dw3 = viewer.window.add_dock_widget(segment_cells, area='right', name='Label Cells')
dw3.setMinimumHeight(180);dw3.setMaximumHeight(180)
dw4 = viewer.window.add_dock_widget(segment_nuclei, area='right', name='Label Nuclei')
dw4.setMinimumHeight(160);dw4.setMaximumHeight(160)
dw5 = viewer.window.add_dock_widget(remove_label, area='right', name='Remove Cell Label')
dw5.setMinimumHeight(100);dw5.setMaximumHeight(100)
dw6 = viewer.window.add_dock_widget(merge_labels, area='right', name='Merge Cell Labels')
dw6.setMinimumHeight(120);dw6.setMaximumHeight(120)
napari.run()
