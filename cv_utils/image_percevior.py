from constants import *
from .glee_detector import *
from dataclasses import dataclass

class GLEE_Percevior:
    def __init__(self,
                 glee_config=GLEE_CONFIG_PATH,
                 glee_checkpoint=GLEE_CHECKPOINT_PATH,
                 device = "cuda:0"):
        self.device = device
        self.glee_model = initialize_glee(glee_config,glee_checkpoint,device)
    def perceive(self,image,confidence_threshold=0.25,area_threshold=2500):
        pred_bboxes, pred_masks, pred_class, pred_confidence = glee_segmentation(image,self.glee_model,threshold_select=confidence_threshold,device=self.device)
        try:
            mask_area = np.array([mask.sum() for mask in pred_masks])
            bbox_trust = np.array([(bbox[0] > 20) & (bbox[2] < image.shape[1] - 20) for bbox in pred_bboxes])
            visualization = visualize_segmentation(image,pred_class[(mask_area>area_threshold) & bbox_trust],pred_masks[(mask_area>area_threshold) & bbox_trust])
            return pred_class[(mask_area>area_threshold) & bbox_trust],pred_masks[(mask_area>area_threshold) & bbox_trust],pred_confidence[(mask_area>area_threshold) & bbox_trust],[visualization]
        except:
            return [],[],[],[image]

@dataclass
class GT_Percevior:
    visualize_seg: bool = False
    env_objects: list = None
    target_objs: list = None
    ignore_obj = ['unknown', 'wall', 'ceiling', 'floor', 'stairs', 'beam', 'book']
    def perceive(self,image, seg, area_threshold=0):
        # pred_bboxes, pred_masks, pred_class, pred_confidence = glee_segmentation(image,self.glee_model,threshold_select=confidence_threshold,device=self.device)
        
        # try:
        area_threshold = seg.size * 0.001
        seg = np.squeeze(seg)
        o_ids = [o_id for o_id in np.unique(seg).tolist() if (o_id < len(self.env_objects)) and (self.env_objects[o_id].lower() not in self.ignore_obj) and (self.target_objs is None or self.env_objects[o_id].lower() in self.target_objs)]
        pred_masks = np.array([seg == o_id for o_id in o_ids])
        seg_class = np.array([self.env_objects[o_id] for o_id in o_ids])
        mask_area = np.array([mask.sum() for mask in pred_masks])
        if self.visualize_seg:
            visualization = visualize_segmentation(image,seg_class[(mask_area>area_threshold)],pred_masks[(mask_area>area_threshold)])
            return seg_class[(mask_area>area_threshold)],pred_masks[(mask_area>area_threshold)],np.ones_like(mask_area), [visualization]
        return seg_class[(mask_area>area_threshold)],pred_masks[(mask_area>area_threshold)],np.ones_like(mask_area),[image]
        # except:
            # print(np.unique(seg).tolist(), len(self.env_objects))
            # return [],[],[],[image]
