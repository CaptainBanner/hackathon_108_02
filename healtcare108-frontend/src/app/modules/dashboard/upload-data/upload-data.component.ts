import { Component, OnInit, Input } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-upload-data',
  templateUrl: './upload-data.component.html',
  styleUrls: ['./upload-data.component.css']
})
export class UploadDataComponent implements OnInit {

  @Input() type: string;
  @Input() title: string;

  selectedFile: File = null;
  fileName = null;

  constructor(private http: HttpClient, private snackBar: MatSnackBar) { }

  ngOnInit() {
  }

  onFileChanged(event) {
    this.selectedFile = event.target.files[0] as File;
    this.fileName = this.selectedFile.name;
  }

  onUpload() {
    // upload code goes here
    let uploadData = new FormData();
    uploadData.append(this.type, this.selectedFile);
    console.log(uploadData);
    console.log(this.type);
    this.http.post('http://40.78.132.182/calculation/getimage/', uploadData)
      .subscribe(event => {
        console.log(event); // handle event here
      });
    this.snackBar.open('Submitted Succesfully.', 'close', {
        duration: 10000,
      });

    uploadData = null;

    this.reset();
  }

  reset() {
    this.selectedFile = null;
    this.fileName = null;
  }

}
