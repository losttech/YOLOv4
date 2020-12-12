namespace DetectUI {
    partial class YoloForm {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing) {
            if (disposing && (components != null)) {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent() {
            this.openPic = new System.Windows.Forms.Button();
            this.pictureBox = new System.Windows.Forms.PictureBox();
            this.openPicDialog = new System.Windows.Forms.OpenFileDialog();
            this.openWeightsDirDialog = new System.Windows.Forms.FolderBrowserDialog();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).BeginInit();
            this.SuspendLayout();
            // 
            // openPic
            // 
            this.openPic.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.openPic.AutoSize = true;
            this.openPic.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.openPic.Enabled = false;
            this.openPic.Location = new System.Drawing.Point(12, 408);
            this.openPic.Name = "openPic";
            this.openPic.Size = new System.Drawing.Size(101, 30);
            this.openPic.TabIndex = 0;
            this.openPic.Text = "Open Image";
            this.openPic.UseVisualStyleBackColor = true;
            this.openPic.Click += new System.EventHandler(this.openPic_Click);
            // 
            // pictureBox
            // 
            this.pictureBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.pictureBox.Location = new System.Drawing.Point(1, -1);
            this.pictureBox.Name = "pictureBox";
            this.pictureBox.Size = new System.Drawing.Size(800, 403);
            this.pictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox.TabIndex = 1;
            this.pictureBox.TabStop = false;
            // 
            // openPicDialog
            // 
            this.openPicDialog.Title = "Open Picture";
            // 
            // openWeightsDirDialog
            // 
            this.openWeightsDirDialog.Description = "Select directory with SavedModel";
            this.openWeightsDirDialog.RootFolder = System.Environment.SpecialFolder.MyDocuments;
            this.openWeightsDirDialog.ShowNewFolderButton = false;
            // 
            // YoloForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.pictureBox);
            this.Controls.Add(this.openPic);
            this.Name = "YoloForm";
            this.Text = "YOLO";
            this.Load += new System.EventHandler(this.YoloForm_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button openPic;
        private System.Windows.Forms.PictureBox pictureBox;
        private System.Windows.Forms.OpenFileDialog openPicDialog;
        private System.Windows.Forms.FolderBrowserDialog openWeightsDirDialog;
    }
}

