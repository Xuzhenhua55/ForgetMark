import copy
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer


class GradDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        # If the model is already dispatched via accelerate (device_map/quant),
        # prevent HF Trainer from calling model.to(device)
        ta = kwargs.get("args", None)
        mdl = kwargs.get("model", None)
        try:
            if ta is not None and getattr(ta, "place_model_on_device", True):
                if hasattr(mdl, "hf_device_map") and getattr(mdl, "hf_device_map"):
                    ta.place_model_on_device = False
        except Exception:
            pass

        super().__init__(*args, **kwargs)
        # Accept method-specific hyperparameters via __init__ (passed from method_args)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.retain_loss_type = str(retain_loss_type).upper()

        # Only build ref model when truly needed (saves memory and avoids .to issues)
        self.ref_model = None
        if self.alpha > 0 and self.retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        # Avoid calling .to(device) on a sharded/offloaded model (device_map="auto").
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "NLL":
            retain_outputs = model(**retain_inputs)
            return retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, _ = compute_kl_divergence(self.model, self.ref_model, retain_inputs)
            return kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forget loss (gradient ascent)
        f = inputs["forget"]
        forget_inputs = {
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"],
            "labels": f["labels"],
        }
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        # Retain loss (optional)
        retain_loss = 0.0
        if self.alpha > 0 and "retain" in inputs and inputs["retain"] is not None:
            r = inputs["retain"]
            retain_inputs = {
                "input_ids": r["input_ids"],
                "attention_mask": r["attention_mask"],
                "labels": r["labels"],
            }
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
