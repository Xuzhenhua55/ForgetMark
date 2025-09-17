import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything
import torch
import gc


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        
        # 自定义保存逻辑，避免使用Trainer的save_model
        import os
        output_dir = trainer_args.output_dir
        if hasattr(model, "is_peft_model") and model.is_peft_model:
            # 如果是PEFT模型，使用peft的保存方法
            import json
            from peft.utils import CONFIG_NAME
            
            # 保存适配器配置
            os.makedirs(output_dir, exist_ok=True)
            # 转换任何非JSON可序列化类型
            config_dict = {k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, (dict, str)) else v) 
                          for k, v in model.peft_config.items()}
            
            # 保存PEFT配置
            with open(os.path.join(output_dir, CONFIG_NAME), "w") as f:
                json.dump(config_dict, f, indent=2)
                
            # 保存PEFT权重
            model.save_pretrained(output_dir)
            # 保存原始模型配置
            if hasattr(model, "config") and model.config is not None:
                model.config.save_pretrained(output_dir)
            print(f"PEFT model successfully saved to {output_dir}")
        else:
            # 原始非PEFT模型
            trainer.save_model(output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
