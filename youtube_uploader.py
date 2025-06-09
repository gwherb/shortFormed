import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

class YouTubeUploader:
    def __init__(self, credentials_file="credentials.json"):
        self.credentials_file = credentials_file
        self.token_file = "token.pickle"
        self.scopes = ["https://www.googleapis.com/auth/youtube.upload"]
        self.youtube = self.authenticate()
    
    def authenticate(self):
        """Handle OAuth2 authentication"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.scopes)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('youtube', 'v3', credentials=creds)
    
    def upload_video(self, video_file, title, description="", tags=None, 
                    category_id="22", privacy_status="private"):
        """
        Upload video to YouTube
        
        Args:
            video_file: Path to video file
            title: Video title
            description: Video description
            tags: List of tags
            category_id: YouTube category ID (22 = People & Blogs)
            privacy_status: 'private', 'public', 'unlisted'
        """
        
        if tags is None:
            tags = []
        
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': category_id
            },
            'status': {
                'privacyStatus': privacy_status,
                'selfDeclaredMadeForKids': False
            }
        }
        
        # Create media upload object
        media = MediaFileUpload(
            video_file,
            chunksize=-1,  # Upload in single request
            resumable=True
        )
        
        # Execute upload
        insert_request = self.youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media
        )
        
        response = self.execute_upload(insert_request)
        return response
    
    def execute_upload(self, insert_request):
        """Execute the upload with progress tracking"""
        response = None
        error = None
        retry = 0
        
        while response is None:
            try:
                print("Uploading file...")
                status, response = insert_request.next_chunk()
                if status:
                    print(f"Uploaded {int(status.progress() * 100)}%")
            except Exception as e:
                error = e
                print(f"An error occurred: {e}")
                retry += 1
                if retry > 3:
                    break
        
        if response is not None:
            video_id = response['id']
            print(f"Video uploaded successfully!")
            print(f"Video ID: {video_id}")
            print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            return response
        else:
            print(f"Upload failed: {error}")
            return None

# Usage example
def upload_example():
    uploader = YouTubeUploader("path/to/credentials.json")
    
    result = uploader.upload_video(
        video_file="my_video.mp4",
        title="My Awesome Video",
        description="This is a test upload using Python",
        tags=["python", "youtube", "automation"],
        privacy_status="private"  # Start with private for testing
    )
    
    return result