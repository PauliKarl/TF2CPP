$ python create_tf_record.py --root_dir=datasets/trainval --label_map_path=sample_class.pbtxt --output_path=my_save_path

$ python model_main.py --model_dir=D:/tf_obj/training --pipeline_config_path=D:/tf_obj/configs/mask_rcnn_inception_v2_coco.config


生成PB文件：在object_detection/目录下运行
 python export_inference_graph.py --input_type image_tensor --pipeline_config_path G:/tf_obj/configs/mask_rcnn_inception_v2_coco.config --trained_checkpoint_prefix G:/tf_obj/training/model.ckpt-0 --output_directory G:/tf_obj/training/output_inference


生成graph.pbtxt文件：在opencv/sources/samples/dnn/目录下运行
 python tf_text_graph_mask_rcnn.py --input G:/tf_obj/training/output_inference/frozen_inference_graph.pb --config G:/tf_obj/training/output_inference/pipeline.config --output G:/tf_obj/training/output_inference/graph.pbtxt
