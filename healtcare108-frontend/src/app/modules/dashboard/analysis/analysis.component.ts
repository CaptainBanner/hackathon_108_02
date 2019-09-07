import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-analysis',
  templateUrl: './analysis.component.html',
  styleUrls: ['./analysis.component.css']
})
export class AnalysisComponent implements OnInit {

  // panelOpenState = false;

  constructor() { }

  ngOnInit() {
  }

  panelOpenState(type: string) {
    console.log(type);
  }

}
