// Based on https://github.com/executablebooks/sphinx-tabs/blob/master/sphinx_tabs/static/tabs.js
// Copyright (c) 2017 djungelorm
// MIT Licensed

function deselectTabset(target) {
  const parent = target.parentNode;
  const grandparent = parent.parentNode;

  if (parent.parentNode.parentNode.getAttribute("id").startsWith("installation")) {

    // Hide all tabs in current tablist, but not nested
    Array.from(parent.children).forEach(t => {
      if (t.getAttribute("name") !== target.getAttribute("name")) {
        t.setAttribute("aria-selected", "false");
      }
    });

    // Hide all associated panels
    Array.from(grandparent.children).slice(1).forEach(p => {  // Skip tablist
      if (p.getAttribute("name") !== target.getAttribute("name")) {
        p.setAttribute("hidden", "false")
      }
    });
  }

  else {
    // Hide all tabs in current tablist, but not nested
    Array.from(parent.children).forEach(t => {
      t.setAttribute("aria-selected", "false");
    });

    // Hide all associated panels
    Array.from(grandparent.children).slice(1).forEach(p => {  // Skip tablist
      p.setAttribute("hidden", "true")
    });
  }

}

// Compatibility with sphinx-tabs 2.1.0 and later
function deselectTabList(tab) {deselectTabset(tab)}
