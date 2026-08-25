(() => {
  "use strict";

  const script = document.currentScript;
  const dataElement = script.previousElementSibling;
  const root = dataElement.previousElementSibling.previousElementSibling;
  const bootstrap = JSON.parse(dataElement.textContent);
  const invokeCommand = async (command, parameters) => {
    const response = await google.colab.kernel.invokeFunction(
      bootstrap.callbackName,
      [command, parameters],
      {},
    );
    return response.data["application/json"];
  };
  mountHumanArena(root, bootstrap.view, invokeCommand, "Colab");
})();
