window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

//Load specific LaTeX packages
document.addEventListener("MathJax.typesetReady", function() {
    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);

    // load LateX package
    MathJax.texConfig({
	packages: ['[tx]amsmath', '[tx]amsfonts', '[tx]amssymb', '[tx]xcolor']
    });

    // Re-typeset after packages load
    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
});

document$.subscribe(() => { 
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})
