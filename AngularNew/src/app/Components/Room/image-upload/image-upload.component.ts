import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, EventEmitter, OnInit, Output,NgZone  } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AnalyzeService } from '../../../Service/analyze.service';
import { Router } from '@angular/router';


@Component({
  selector: 'app-image-upload',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './image-upload.component.html',
  styleUrl: './image-upload.component.css'
})
export class ImageUploadComponent {
  selectedFile!: File;  
  imagePreview: string | null = null;
  selectedBoxVolume: number  = 0; 
  error: string | null = null;
  loading: boolean = false; 
  @Output() analysisCompleted = new EventEmitter<any[]>(); // שליחת התוצאה למקשר
  @Output() capacity = new EventEmitter<number>();  // שליחה של הנפח

  constructor(private http: HttpClient,private imageAnalysisService:AnalyzeService,private router:Router,private ngZone:NgZone) { }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
  
    //בדיקת סוג קובץ -מותר רק תמונה
      if (!file.type.startsWith('image/')) {
        this.error = 'Please select a valid image file.';
        this.imagePreview = null;
        return;
      }
  
      this.error = null;
      this.selectedFile = file;
  
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        this.imagePreview = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  }
  
  onSubmit(): void {
    if (!this.selectedFile) {
      this.error = 'Please select an image before submitting.';
      return;
    }

    this.loading = true;
     
    this.imageAnalysisService.imageAnalysis(this.selectedFile).subscribe({
      next: (response) => {
        this.ngZone.run(() => {
          this.analysisCompleted.emit(response);
          this.capacity.emit(this.selectedBoxVolume);
        });
        this.loading=false;
      },
      error: (error) => {
        console.error('שגיאה בניתוח תמונה:', error);
        this.loading = false;
      }
    });
  }

goBack(): void {
  this.router.navigate(['/room-selection'])
}
}
  
 
 

