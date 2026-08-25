export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "sapai-human-output";
    el.replaceChildren(root);
    let requestSequence = 0;
    const invokeCommand = (command, parameters) => new Promise((resolve) => {
      requestSequence += 1;
      const id = `${Date.now()}-${requestSequence}`;
      const receive = () => {
        const response = model.get("response") || {};
        if (response.id !== id) return;
        model.off("change:response", receive);
        resolve(model.get("view"));
      };
      model.on("change:response", receive);
      model.set("request", {id, command, parameters});
      model.save_changes();
    });
    return mountHumanArena(root, model.get("view"), invokeCommand, "Kaggle");
  },
};
