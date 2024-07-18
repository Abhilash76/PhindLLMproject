function define(
  unknown: string[],
  any: (Jupyter: {
    notebook: {
      get_selected_cell: () => any;
      get_selected_index: () => any;
      insert_cell_at_index: (arg0: string, arg1: any) => any;
      scroll_to_cell: (arg0: any, arg1: number) => void;
      execute_cell_and_select_below: (arg0: any) => void;
    };
    toolbar: {
      add_buttons_group: (
        arg0: {
          label: string;
          icon: string;
          callback: () => void;
          id: string;
        }[]
      ) => void;
    };
  }) => { load_ipython_extension: () => void }
  // eslint-disable-next-line @typescript-eslint/no-empty-function
) {}

define(['base/js/namespace'], (Jupyter: {
  notebook: {
    get_selected_cell: () => any;
    get_selected_index: () => any;
    insert_cell_at_index: (arg0: string, arg1: any) => any;
    scroll_to_cell: (arg0: any, arg1: number) => void;
    execute_cell_and_select_below: (arg0: any) => void;
  };
  toolbar: {
    add_buttons_group: (
      arg0: { label: string; icon: string; callback: () => void; id: string }[]
    ) => void;
  };
}) => {
  function retrieveCellContents() {
    // Get selected cell
    const selectedCell = Jupyter.notebook.get_selected_cell();

    // Check if the selected cell is a code cell
    let dummy_markdown;
    if (selectedCell && selectedCell.cell_type === 'code') {
      // Extract content of the selected code cell
      const codeContent = selectedCell.get_text();

      // Create a new Markdown cell above the selected code cell
      const index = Jupyter.notebook.get_selected_index();
      const newCell = Jupyter.notebook.insert_cell_at_index('markdown', index);

      // Set the code content as the text of the new Markdown cell
      dummy_markdown =
        'This code cell provides insights from our data analysis project, covering objectives, methodology, and key findings.We cover data preprocessing, exploratory data analysis (EDA), and model evaluation. ';
      newCell.set_text(dummy_markdown, codeContent);

      // Deselect the code cell after adding the new Markdown cell
      selectedCell.unselect();

      // Scroll notebook to the top where the new cell is inserted
      Jupyter.notebook.scroll_to_cell(index, 0);

      // Run the newly created Markdown cell
      Jupyter.notebook.execute_cell_and_select_below(index);
    }
  }

  function load_extension() {
    // Create a button and add it to the toolbar
    Jupyter.toolbar.add_buttons_group([
      {
        label: 'Create Markdown',
        icon: 'fa-code',
        callback: retrieveCellContents,
        id: 'retrieve-cell-contents-button'
      }
    ]);
  }

  return {
    load_ipython_extension: load_extension
  };
});
