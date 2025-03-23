import { Component, EventEmitter, Input, Output } from '@angular/core';
import { AuthService } from '../../../Service/auth.service';
import { RoomService } from '../../../Service/room.service';
import { RoomPost } from '../../../Models/RoomPost.model';
import { ReactiveFormsModule, FormGroup, FormControl, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-add-room',
  standalone: true,
  imports: [CommonModule,ReactiveFormsModule],
  templateUrl: './add-room.component.html',
  styleUrl: './add-room.component.css'
})
export class AddRoomComponent {
  @Input() categoryId!: number;
  @Input() showModal: boolean = false; // if showModal is true
  @Output() roomCreated = new EventEmitter<any>();
  @Output() closeModalBool = new EventEmitter<void>(); // close popup modal

  roomForm: FormGroup;
  isModalOpen: boolean = false; // state of modal
  
  constructor(private authService: AuthService, private roomService: RoomService) {
    this.roomForm = new FormGroup({
      roomName: new FormControl('', Validators.required) 
    });
  }

  openModal() {
    this.isModalOpen = true; 
  }

  closeModal() {
    this.isModalOpen = false; 
  }

  addRoom() {
    if (this.roomForm.valid) {
      const roomData: RoomPost = {
        name: this.roomForm.value.roomName,
        categoryId: this.categoryId,
        userId: this.authService.getCurrentUserId(),
        user: this.authService.getCurrentUser()
      };

      this.roomService.addServer(roomData).subscribe({
        next: (newRoom) => {
          console.log(newRoom);
          this.roomCreated.emit(newRoom);
          this.closeModal(); 
        },
        error: (error) => {
          console.error('Failed to create room:', error);
        }
      });
    }
  }
}

