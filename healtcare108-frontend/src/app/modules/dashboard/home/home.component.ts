import { Component, OnInit } from '@angular/core';
import { MatTabChangeEvent } from '@angular/material/tabs';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {

  type: string;

  constructor() { }

  ngOnInit() {
  }

  onTabSelection(event: MatTabChangeEvent) {
    this.type = event.tab.textLabel;
  }

}
