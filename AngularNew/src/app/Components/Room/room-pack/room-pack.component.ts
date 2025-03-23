import { Component, EventEmitter, Output } from '@angular/core';
import { ImageUploadComponent } from '../image-upload/image-upload.component'; 
import { ResultComponent } from '../result/result.component'; 
import { CommonModule } from '@angular/common';
import { AddRoomComponent } from '../add-room/add-room.component';
import { ImageAnalsys } from '../../../Models/ImageAnalsys.model';
import { Room } from '../../../Models/Room.model';
import { ActivatedRoute } from '@angular/router';
@Component({
  selector: 'app-room-pack',
  standalone: true,
  imports: [CommonModule, ImageUploadComponent, ResultComponent,AddRoomComponent], 
  templateUrl: './room-pack.component.html',
  styleUrl: './room-pack.component.css'
})
export class RoomPackComponent {
  imageAnalysisResults: ImageAnalsys[] = [];
  selectedBoxVolume: number =0;  
  createdRoom!: Room;
  categoryId:number=0;
  showAddRoomPopup :boolean= true; 
  showImageUpload :boolean= false;
  showResults :boolean= false;
  @Output() roomCreated = new EventEmitter<any>();
  @Output() closeModal = new EventEmitter<void>();
  constructor(private route: ActivatedRoute) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      const id = params.get('roomId');
      if (id) {
        this.categoryId = +id; 
      }
    });
  }

  handleRoomCreated(room: Room) {
    this.createdRoom = room;
    this.showAddRoomPopup = false;
    this.showImageUpload = true;
  }
  openAddRoomPopup() {
    this.showAddRoomPopup = true;
  }
  closeAddRoomPopup() {
    this.showAddRoomPopup = false;
  }

  handleAnalysisResults(results: ImageAnalsys[]) {
    this.imageAnalysisResults = results; 
    this.showImageUpload = false;
    this.showResults = true;
  }
  handleCapacity(capacity:number){
    this.selectedBoxVolume=capacity;
  }
  goBack() {
    this.showResults = false;
    this.showImageUpload = true;
  }
}
